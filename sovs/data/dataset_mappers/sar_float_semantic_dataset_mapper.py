# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for SAR float/TIF semantic segmentation in SOVS-Net.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from osgeo import gdal
import random

__all__ = ["SARFloatSemanticDatasetMapper", "SARFloatSemanticDatasetMapperVal"]


class RandomScaleThenCrop(T.Augmentation):
    def __init__(self, scale_range=(1.2, 1.5), crop_size=(256, 256), prob=0.5):
        super().__init__()
        self.scale_range = scale_range
        self.crop_size = crop_size
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()

        h, w = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)

        resize_transform = T.ResizeTransform(h, w, new_h, new_w)
        dummy_image = np.zeros((new_h, new_w, 3), dtype=np.float32)
        crop_transform = T.RandomCrop("absolute", self.crop_size).get_transform(dummy_image)
        return T.TransformList([resize_transform, crop_transform])


class FuncTransform(T.Transform):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self._set_attributes()

    def apply_image(self, img):
        return self.func(img)

    def apply_segmentation(self, seg):
        return seg

    def apply_coords(self, coords):
        return coords


class RandomGamma(T.Augmentation):
    def __init__(self, gamma_range=(0.9, 1.1), prob=0.5):
        super().__init__()
        self.gamma_range, self.prob = gamma_range, prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()

        gamma = random.uniform(*self.gamma_range)
        def _gamma(img, g=gamma):
            img = np.clip(img, 0.0, 1.0).astype(img.dtype)
            out = np.power(img, g)
            return np.clip(out, 0.0, 1.0).astype(img.dtype)
        return FuncTransform(_gamma)


class RadiometricScaling(T.Augmentation):
    def __init__(self, scale_range=(0.8, 1.2), prob=0.5):
        super().__init__()
        self.scale_range, self.prob = scale_range, prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()
        factor = random.uniform(*self.scale_range)
        def _scale(img, f=factor):
            return np.clip(img * f, 0.0, 1.0).astype(img.dtype)
        return FuncTransform(_scale)


class SpeckleJitter(T.Augmentation):
    def __init__(self, std_range=(0.01, 0.05), prob=0.5):
        super().__init__()
        self.std_range, self.prob = std_range, prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()
        sigma = random.uniform(*self.std_range)
        noise = np.random.normal(0, sigma, image.shape).astype(image.dtype)
        def _speckle(img, n=noise):
            return np.clip(img + img * n, 0.0, 1.0).astype(img.dtype)
        return FuncTransform(_speckle)


class RandomBandDropout(T.Augmentation):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()

        if np.array_equal(image[:, :, 0], image[:, :, 1]):
            return T.NoOpTransform()

        band_to_keep = random.randint(0, 1)

        def _dropout(img, band=band_to_keep):
            out_img = np.zeros_like(img)

            if band == 0:
                channel_data = img[:, :, 0]
            else:
                channel_data = img[:, :, 1]

            out_img[:, :, 0] = channel_data
            out_img[:, :, 1] = channel_data
            out_img[:, :, 2] = channel_data

            return np.clip(out_img, -np.inf, 1.0).astype(img.dtype)

        return FuncTransform(_dropout)

class SARFloatSemanticDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = []

        # Add random scale and crop after initial resize & crop
        augs.append(RandomScaleThenCrop(
            scale_range=(1.2, 1.5),
            crop_size=cfg.INPUT.CROP.SIZE,
            prob=0.1
        ))

        augs.append(RadiometricScaling(prob=0.1))
        augs.append(SpeckleJitter(prob=0.1))
        augs.append(RandomBandDropout(prob=0.1))

        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = ReadTif(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=0).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict

def ReadTif(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError(f"Failed to open SAR image: {filename}")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    channels = dataset.RasterCount

    # Use the first band of GDAL to determine the data type
    band0 = dataset.GetRasterBand(1)
    dt = band0.DataType
    type_name = gdal.GetDataTypeName(dt)

    # Only Byte (int8) or Float types are allowed
    if type_name not in ['Byte', 'Float32', 'Float64']:
        raise ValueError(f"Unsupported GDAL data type: {type_name}")

    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    dataset = None

    if type_name in ['Float32', 'Float64']:
        # Process floating-point SAR data.
        if channels == 1:#for single band
            GdalImg_data = (GdalImg_data + 25.0) / 30.0
            GdalImg_data[GdalImg_data > 1] = 1
            GdalImg_data = np.stack([GdalImg_data]*3, axis=-1)  # (224, 224, 3)
        else:#  channels == 2 VVforband0 VHforband1
            channel1 = (GdalImg_data[0] + 25.0) / 25.0
            channel2 = (GdalImg_data[1] + 32.5) / 25.0
            channel1[channel1 > 1] = 1
            channel2[channel2 > 1] = 1
            channel3 = (channel1 + channel2) / 2.0
            GdalImg_data = np.stack([channel1, channel2, channel3], axis=2)
    elif type_name == 'Byte':
        # int8 type (0-255) processing: First normalized to [0,1]
        GdalImg_data = GdalImg_data / 255.0
        if channels == 1:
            GdalImg_data = np.stack([GdalImg_data]*3, axis=-1)  # (224, 224, 3)
        else:
            channel1 = GdalImg_data[0]
            channel2 = GdalImg_data[1]
            channel3 = (channel1 + channel2) / 2.0
            GdalImg_data = np.stack([channel1, channel2, channel3], axis=2)
    else:
        raise ValueError(f"Unhandled GDAL data type: {type_name}")

    return GdalImg_data.astype(np.float32)


class SARFloatSemanticDatasetMapperVal:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=False,
        *,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        #self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] No Augmentations used in {mode}")

    @classmethod
    def from_config(cls, cfg, is_train=False):
        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert not self.is_train, "SARFloatSemanticDatasetMapperVal should only be used for Testing!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = ReadTif(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=0).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
