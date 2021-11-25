
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from nuscenes.utils.geometry_utils import  BoxVisibility
from vis_det.model_pool.kitti_dataset import kitti_object
import os

def get_kitti_dicts(path="./", categories=None):
    """
    This is a helper fuction that create dicts from KITTI to detectron2 format.
    KITTI annotation use 3d bounding box, but for detectron we need 2d bounding box.
    The simplest solution is get max x, min x, max y and min y coordinates from 3d bb and
    create 2d box. So we lost accuracy, but this is not critical.
    :param path: <string>. Path to KITTI dataset.
    :param categories <list<string>>. List of selected categories for detection.
        Categories names:
    :return: <dict>. Return dict with data annotation in detectron2 format.
    """
    assert(path[-1] == "/"), "Insert '/' in the end of path"
    kitti_dataset = kitti_object(path)
    print(kitti_dataset)

    categories = {'Car':2, 'Bicycle':1, 'Pedestrian':0}

    dataset_dicts = []
    idx = 0
    for idx in tqdm(range(0, 10)):#len(kitti_dataset))):
        
        image = kitti_dataset.get_image(idx)
        # print(image.shape)

        record = {}
        record["file_name"] = os.path.join(kitti_dataset.image_dir, "%06d.png" % (idx))
        record["image_id"] = idx
        record["height"] = image.shape[0]
        record["width"] = image.shape[1]

        boxes = kitti_dataset.get_label_objects(idx)

        # Go through all bounding boxes
        objs = []
        for box in boxes:
            if box.type not in ['Car', 'Bicycle', 'Pedestrian']:
                continue
            obj = {
                "bbox": (int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": categories[box.type],
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)


    return dataset_dicts