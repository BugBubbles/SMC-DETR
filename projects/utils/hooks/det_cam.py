from torchcam.methods import GradCAM
from torch import Tensor, nn
from typing import Union, List, Optional, Any
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange


class DetGradCAM(GradCAM):
    def __init__(self, *args, relu: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._relu = relu

    def __call__(
        self, loss: Tensor, normalized: bool = True, **kwargs: Any
    ) -> List[Tensor]:
        # Integrity check
        assert loss.requires_grad, "Loss must require gradients"
        # Compute CAM
        return self.compute_cams(loss, normalized, **kwargs)

    def _backprop(self, loss: Tensor, retain_graph: bool = False) -> None:
        """Backpropagate the loss for a specific output class"""
        # Backpropagate to get the gradients on the hooked layer
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)

    def compute_cams(
        self,
        loss: Tensor,
        normalized: bool = True,
        permute: str = "b n c",
        **kwargs: Any,
    ) -> List[Tensor]:
        """Compute the CAM for a specific output class.

        Args:
            class_idx: the class index of the class to compute the CAM of, or a list of class indices. If it is a list,
                the list needs to have valid class indices and have a length equal to the batch size.
            scores: forward output scores of the hooked model of shape (N, K)
            normalized: whether the CAM should be normalized
            kwargs: keyword args of `_get_weights` method

        Returns:
            list of class activation maps of shape (N, H, W), one for each hooked layer. If a list of class indices
                was passed to arg `class_idx`, the k-th element along the batch axis will be the activation map for
                the k-th element of the input batch for class index equal to the k-th element of `class_idx`.
        """
        # Get map weight & unsqueeze it
        weights = self._get_weights(loss, permute, **kwargs)

        cams: List[Tensor] = []

        hook_a = (rearrange(x, permute) for x in self.hook_a)

        with torch.no_grad():
            for weight, activation in zip(weights, hook_a):
                weight = weight.unsqueeze(1)

                # Perform the weighted combination to get the CAM
                cam = torch.nansum(weight * activation, dim=-1)

                if self._relu:
                    cam = F.relu(cam, inplace=True)

                # Normalize the CAM
                if normalized:
                    cam = self._normalize(cam)

                cams.append(cam)

        return cams

    def _get_weights(self, loss: Tensor, permute, **kwargs: Any) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        # Backpropagate
        self._backprop(loss, **kwargs)

        self.hook_g: List[Tensor]  # type: ignore[assignment]
        # Global average pool the gradients over spatial dimensions
        hook_g = (rearrange(x, permute) for x in self.hook_g)

        return [grad.flatten(2).mean(-2) for grad in hook_g]


def visualize_features(
    heat_map: Union[List[Tensor], Tensor],
    spatial_shapes,
    batch_data_samples,
    images_dir="./images",
):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if isinstance(heat_map, Tensor):
        heat_map = heat_map.split([h * w for h, w in spatial_shapes])
    else:
        assert isinstance(heat_map, list)
    image = plt.imread(batch_data_samples[0].img_path)
    input_shape = batch_data_samples[0].batch_input_shape
    img_shape = batch_data_samples[0].img_shape
    image_name = batch_data_samples[0].img_path.split("/")[-1].replace(".png", "")
    # plt.cla()
    # plt.imshow(image)
    # plt.savefig(f"images.png", bbox_inches="tight", pad_inches=0)
    for heat, (h, w) in zip(heat_map, spatial_shapes):
        plt.cla()
        plt.imshow(image)
        heat = heat.view(h, w)
        heat = F.interpolate(heat[None, None], input_shape).squeeze()
        heat = heat[0 : img_shape[0], 0 : img_shape[1]]
        heat = F.interpolate(heat[None, None], image.shape[:2]).squeeze()
        plt.imshow(heat.cpu().numpy(), cmap="jet", alpha=0.2)
        plt.axis("off")  # 可选，去掉坐标轴
        plt.savefig(f"images/{image_name}_{h}.png", bbox_inches="tight", pad_inches=0)
