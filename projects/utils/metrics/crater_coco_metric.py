# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import List, Optional, Sequence, Union
import numpy as np
from mmengine.fileio import load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
from mmdet.evaluation import CocoMetric
from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
import tqdm


@METRICS.register_module()
class CraterCocoMetric(CocoMetric):
    def __init__(
        self,
        ann_file: Optional[str] = None,
        metric: Union[str, List[str]] = "bbox",
        classwise: bool = False,
        proposal_nums: Sequence[int] = (100, 300, 1000),
        iou_thrs: Optional[Union[float, Sequence[float]]] = None,
        metric_items: Optional[Sequence[str]] = None,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        file_client_args: dict = None,
        backend_args: dict = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        sort_categories: bool = False,
        use_mp_eval: bool = False,
        score_thr: Optional[Union[float, Sequence[float]]] = (0.5, 1.0, 0.05),
        precision_thr: float = 0.5,
        recall_thr: float = 0.5,
    ):
        super().__init__(
            ann_file,
            metric,
            classwise,
            proposal_nums,
            iou_thrs,
            metric_items,
            format_only,
            outfile_prefix,
            file_client_args,
            backend_args,
            collect_device,
            prefix,
            sort_categories,
            use_mp_eval,
        )
        self.score_thr = (
            [score_thr] if isinstance(score_thr, float) else np.arange(*score_thr)
        )
        self.precision_thr = precision_thr
        self.recall_thr = recall_thr

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info("Converting ground truth to coco format...")
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix
            )
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta["classes"]
            )
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info("results are saved in " f"{osp.dirname(outfile_prefix)}")
            return eval_results

        for metric in self.metrics:
            logger.info(f"Evaluating {metric}...")

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == "proposal_fast":
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger
                )
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]
                    log_msg.append(f"\nAR@{num}\t{ar[i]:.4f}")
                log_msg = "".join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = "bbox" if metric == "proposal" else metric
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")
            try:
                predictions = load(result_files[metric])
                if iou_type == "segm":
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop("bbox")
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error("The testing results of the whole dataset is empty.")
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                "mAP": 0,
                "mAP_50": 1,
                "mAP_75": 2,
                "mAP_s": 3,
                "mAP_m": 4,
                "mAP_l": 5,
                "AR@100": 6,
                "AR@300": 7,
                "AR@1000": 8,
                "AR_s@1000": 9,
                "AR_m@1000": 10,
                "AR_l@1000": 11,
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(f'metric item "{metric_item}" is not supported')

            if metric == "proposal":
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        "AR@100",
                        "AR@300",
                        "AR@1000",
                        "AR_s@1000",
                        "AR_m@1000",
                        "AR_l@1000",
                    ]

                for item in metric_items:
                    val = float(f"{coco_eval.stats[coco_metric_names[item]]:.3f}")
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval["precision"]
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float("nan")
                        t.append(f'{nm["name"]}')
                        t.append(f"{round(ap, 3)}")
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float("nan")
                            t.append(f"{round(ap, 3)}")

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float("nan")
                            t.append(f"{round(ap, 3)}")
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = [
                        "category",
                        "mAP",
                        "mAP_50",
                        "mAP_75",
                        "mAP_s",
                        "mAP_m",
                        "mAP_l",
                    ]
                    results_2d = itertools.zip_longest(
                        *[results_flatten[i::num_columns] for i in range(num_columns)]
                    )
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info("\n" + table.table)

                if metric_items is None:
                    metric_items = ["mAP", "mAP_50", "mAP_s", "AR_s@1000"]

                for metric_item in metric_items:
                    key = f"{metric}_{metric_item}"
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f"{round(val, 3)}")

                ap = coco_eval.stats[:6]
                logger.info(
                    f"{metric}_mAP_copypaste: {ap[0]:.3f} "
                    f"{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} "
                    f"{ap[4]:.3f} {ap[5]:.3f}"
                )
                # calculate Precision and Recall
                precisions = []
                recalls = []
                for score_thr in tqdm.tqdm(
                    self.score_thr, desc="Calculating Precision and Recall"
                ):
                    tps = []
                    fps = []
                    fns = []
                    for evalImg in coco_eval.evalImgs:
                        idx = np.greater(evalImg["dtScores"], score_thr)
                        dt_matches = evalImg["dtMatches"][0][idx]
                        gt_matches = evalImg["gtMatches"][0]
                        dt_ignore = evalImg["dtIgnore"][0][idx]
                        # gtIgnore 形状通常为 [T, K, A] 或 [T, R, K, A]，视 COCO 版本而定
                        gt_ignore = evalImg["gtIgnore"]

                        # 将所有类别、召回点展开后统计
                        tps.append(np.logical_and(dt_matches > 0, dt_ignore == 0).sum())
                        fps.append(
                            np.logical_and(dt_matches == 0, dt_ignore == 0).sum()
                        )
                        fns.append(
                            np.logical_and(gt_matches == 0, gt_ignore == 0).sum()
                        )

                    precisions.append(sum(tps) / (sum(tps) + sum(fps) + 1e-6))
                    recalls.append(sum(tps) / (sum(tps) + sum(fns) + 1e-6))
                eval_results[f"{metric}_precision"] = precisions[0]
                eval_results[f"{metric}_recall"] = recalls[0]
                # 取F1最大值对应的 Precision 和 Recall
                # precisions = np.array(precisions)
                # recalls = np.array(recalls)
                # best_idx = np.argmax(precisions * recalls)
                # if (
                #     precisions[best_idx] > self.precision_thr
                #     and recalls[best_idx] > self.recall_thr
                # ):
                #     eval_results[f"{metric}_precision"] = precisions[best_idx]
                #     eval_results[f"{metric}_recall"] = recalls[best_idx]
                # else:
                #     best_idx = np.argmax(
                #         precisions * recalls / (precisions + recalls + 1e-6)
                #     )
                #     if (
                #         precisions[best_idx] > self.precision_thr
                #         and recalls[best_idx] > self.recall_thr
                #     ):
                #         eval_results[f"{metric}_precision"] = precisions[best_idx]
                #         eval_results[f"{metric}_recall"] = recalls[best_idx]
                #     else:
                #         best_idx = precisions.shape[0] // 2
                #         eval_results[f"{metric}_precision"] = precisions[best_idx]
                #         eval_results[f"{metric}_recall"] = recalls[best_idx]
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@METRICS.register_module()
class CraterCocoMetricByDiameter(CocoMetric):
    def __init__(
        self,
        ann_file: Optional[str] = None,
        metric: Union[str, List[str]] = "bbox",
        classwise: bool = False,
        proposal_nums: Sequence[int] = (100, 300, 1000),
        iou_thrs: Optional[Union[float, Sequence[float]]] = None,
        metric_items: Optional[Sequence[str]] = None,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        file_client_args: dict = None,
        backend_args: dict = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        sort_categories: bool = False,
        use_mp_eval: bool = False,
        score_thr: Optional[Union[float, Sequence[float]]] = (0.5, 1.0, 0.05),
        diameter_step: float = 50,
    ):
        super().__init__(
            ann_file,
            metric,
            classwise,
            proposal_nums,
            iou_thrs,
            metric_items,
            format_only,
            outfile_prefix,
            file_client_args,
            backend_args,
            collect_device,
            prefix,
            sort_categories,
            use_mp_eval,
        )
        self.score_thr = (
            [score_thr] if isinstance(score_thr, float) else np.arange(*score_thr)
        )
        self.diameter_step = diameter_step

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info("Converting ground truth to coco format...")
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix
            )
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta["classes"]
            )
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info("results are saved in " f"{osp.dirname(outfile_prefix)}")
            return eval_results

        for metric in self.metrics:
            logger.info(f"Evaluating {metric}...")

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == "proposal_fast":
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger
                )
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]
                    log_msg.append(f"\nAR@{num}\t{ar[i]:.4f}")
                log_msg = "".join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = "bbox" if metric == "proposal" else metric
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")
            try:
                predictions = load(result_files[metric])
                if iou_type == "segm":
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop("bbox")
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error("The testing results of the whole dataset is empty.")
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOevalByDiameter(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            # coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                "mAP": 0,
                "mAP_50": 1,
                "mAP_75": 2,
                "mAP_s": 3,
                "mAP_m": 4,
                "mAP_l": 5,
                "AR@100": 6,
                "AR@300": 7,
                "AR@1000": 8,
                "AR_s@1000": 9,
                "AR_m@1000": 10,
                "AR_l@1000": 11,
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(f'metric item "{metric_item}" is not supported')

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return {
            lbl: coco_eval.stats[it]
            for it, lbl in enumerate(coco_eval.params.areaRngLbl)
        }


class COCOevalByDiameter(COCOeval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # a = np.concatenate([np.arange(0, 100, 10), [np.inf]])
        # b = np.zeros_like(a)
        a = (
            np.concatenate(
                [
                    [0, 10],
                    np.arange(12, 20, 2),
                    np.arange(20, 40, 5),
                    np.arange(40, 80, 10),
                ]
            )
            / 1.792
        )
        b = np.concatenate([a[1:], [np.inf]])
        self.params.areaRng = np.vstack([a, b]).T ** 2
        self.params.areaRngLbl = [f"{x.item()}-{y.item()}" for x, y in zip(a, b)]
        self.params.iouThrs = np.arange(0.1, 0.95, 0.05)

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = "@[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] AP = {:0.3f}, AR = {:0.3f}"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # dimension of precision: [TxRxKxAxM]
            p = self.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                p = p[t]
            p = p[:, :, :, aind, mind]
            # dimension of recall: [TxKxAxM]
            r = self.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                r = r[t]
            r = r[:, :, aind, mind]

            ps = []
            rs = []
            for _p, _r in zip(p, r):
                _p = _p[_p > -1]
                _r = _r[_r > -1]
                if len(_p) == 0:
                    ps.append(-1)
                else:
                    ps.append(_p.mean())
                if len(_r) == 0:
                    rs.append(-1)
                else:
                    rs.append(_r.mean())
            ps = np.array(ps)
            rs = np.array(rs)
            ind = np.argmax(ps * rs)
            mean_p = ps[ind]
            mean_r = rs[ind]
            # if len(r[r > 0]) == 0:
            #     mean_r = -1
            # else:
            #     mean_r = r[r > 0].mean()
            # if len(p[p > 0]) == 0:
            #     mean_p = -1
            # else:
            #     mean_p = p[p > 0].mean()
            print(iStr.format(iouStr, areaRng, maxDets, mean_p, mean_r))
            with open("test.csv", "a") as f:
                f.write(f"{mean_p},{mean_r}\n")
            return (mean_p, mean_r)

        def _summarizeDets():
            stats = []
            for it, areaRngLbl in enumerate(self.params.areaRngLbl):
                stats.append(_summarize(1, areaRng=areaRngLbl))
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = (
            self.ious[imgId, catId][:, gtind]
            if len(self.ious[imgId, catId]) > 0
            else self.ious[imgId, catId]
        )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape(
            (1, len(dt))
        )
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }
