# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for SOVS-Net meta-architecture.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList


@META_ARCH_REGISTRY.register()
class SOVSNet(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        in_features,
        fast_inference: bool,
        #ignore_prompt_ids,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "conv":
                    params.requires_grad = True if "conv" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                elif clip_finetune == "mlp":
                    params.requires_grad = True if "mlp" in name or "position" in name else False
                elif clip_finetune == "full_res5":
                    if "stages.3" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        if clip_finetune == "fast_infer":
            for name, params in self.sem_seg_head.predictor.transformer.named_parameters():
                if "head1" in name or "head2" in name or "head0" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
        finetune_backbone = backbone_multiplier > 0.
        for name, params in self.backbone.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

        self.sliding_window = sliding_window

        self.clip_resolution = (256, 256)
        self.sequential = False
        del self.backbone
        self.in_features = in_features
        self.fast_inference = fast_inference
        self.clip_finetune = clip_finetune
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        return {
            #"ignore_prompt_ids": cfg.MODEL.SEM_SEG_HEAD.IGNORE_PROMPT_IDS,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "in_features": cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES,
            "fast_inference": cfg.TEST.FAST_INFERENCE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        self.sliding_window = False

        size_divisibility = -1 if not self.training else self.size_divisibility

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images,size_divisibility)

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility)


        clip_images = clip_images.tensor
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)


        clip_vis_dense = clip_features["clip_vis_dense"]

        fusion_features = {k: v.clone() for k,v in clip_features.items() if k in self.in_features}
        outputs = self.sem_seg_head(clip_vis_dense, fusion_features)
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            num_classes = outputs[0].shape[1]
            mask = targets != self.sem_seg_head.ignore_value
            losses = {}
            aux_weights = [1.0, 0.25, 0.25, 0.25, 0.25]
            for i, output_ in enumerate(outputs):
                if self.clip_finetune == "fast_infer" and i==0:
                    continue
                currentweight = aux_weights[i] if i < len(aux_weights) else 1.0
                output_size = output_.shape[-2:]
                target_size = targets.shape[-2:]

                if output_size != target_size:
                    target_down = F.interpolate(targets.float().unsqueeze(1), size=output_size, mode="nearest").squeeze(
                        1).long()
                    mask_down = F.interpolate(mask.float().unsqueeze(1), size=output_size, mode="nearest").squeeze(
                        1).bool()
                else:
                    target_down = targets
                    mask_down = mask
                _onehot = F.one_hot(target_down[mask_down], num_classes=num_classes).float()

                output_ = output_.permute(0, 2, 3, 1)
                pred_flat = output_[mask_down]

                targ_flat = _onehot
                loss = F.binary_cross_entropy_with_logits(pred_flat, targ_flat) * currentweight

                losses.update({f"loss_sem_seg_{i}" : loss})

            return losses

        else:
            if self.fast_inference:
                outputs = outputs[0]
            else:
                outputs = outputs[0].sigmoid()
            image_size = images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


