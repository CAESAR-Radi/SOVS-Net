import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

def _get_sar11_meta():
    sar11_classes = ["water", "building", "road", "cropland", "forest", "shrubland", "grassland", "wetland", "bareland", "snow and ice", "airport"]

    ret = {
        "stuff_classes" : sar11_classes,
    }
    return ret

def register_sar11(root):
    meta = _get_sar11_meta()

    def get_loader(image_dir, gt_dir):
        return lambda: load_sem_seg(gt_dir, image_dir, gt_ext='tif', image_ext='tif')

    for name, image_dirname, sem_seg_dirname in [
        ("train", r"SAR-OVSeg_release/train/images",
         r"SAR-OVSeg_release/train/annotations"),
        ("val_all5", r"SAR-OVSeg_release/test/images",
         r"SAR-OVSeg_release/test/annotations"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f"sar11_{name}_sem_seg"
        DatasetCatalog.register(full_name, get_loader(image_dir, gt_dir))
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = "./datasets"
register_sar11(_root)
