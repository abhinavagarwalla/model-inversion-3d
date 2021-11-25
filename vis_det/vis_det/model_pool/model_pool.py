import yaml
import os
from vis_det.meta_arch import warp_rcnn, warp_retina
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import detectron2
from vis_det.model_pool.get_kitti import get_kitti_dicts

def load_default_predictor(name):
    """
    Load the default D2 predictor model by calling the model zoo function in D2.

    Args:
    -- name (string): the name of the predictor.

    Returns:
    -- predictor: default predictor loaded by D2.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(name))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


class config(object):
    """
    Configuration object. Get config object from a dict.
    """
    def __init__(self, conf):
        super().__init__()
        for key, value in conf.items():
            setattr(self, key, value)


def get_model(name):
    """
    Get the warpped model given the name. Current support name:
    "COCO-Detection/retinanet_R_50_FPN";
    "COCO-Detection/retinanet_R_101_FPN";

    "COCO-Detection/faster_rcnn_R_50_FPN";
    "COCO-Detection/faster_rcnn_R_101_FPN";

    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN";
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN";

    Args:
    -- name (string): model name.

    Returns:
    -- model: warped model with visualization function.
    -- args: visualization config.
    -- cfg: detector cfg.
    -- predictor: D2 default predictor instances.
    """
    if name == "COCO-Detection/retinanet_R_50_FPN":
        stream = open("./config/retina.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_retina(predictor.model)
    elif name == "COCO-Detection/retinanet_R_101_FPN":
        stream = open("./config/retina.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_retina(predictor.model)
    elif name == "COCO-Detection/faster_rcnn_R_50_FPN":
        stream = open("./config/fasterrcnn.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_rcnn(predictor.model)
    elif name == "COCO-Detection/faster_rcnn_R_101_FPN":
        stream = open("./config/fasterrcnn.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_rcnn(predictor.model)
    elif name == "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN":
        stream = open("./config/maskrcnn.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_rcnn(predictor.model)
    elif name == "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN":
        stream = open("./config/maskrcnn.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_rcnn(predictor.model)
    elif name == "Cityscapes/mask_rcnn_R_50_FPN":
        stream = open("./config/maskrcnn.yaml", 'r')
        args = config(yaml.load(stream, Loader=yaml.FullLoader))
        stream.close()
        name_ = "Cityscapes/mask_rcnn_R_50_FPN.yaml"
        predictor, cfg = load_default_predictor(name_)
        model = warp_rcnn(predictor.model)
    # cfg.MODEL.DEVICE = 'cpu'
    # model.to('cpu')
    model.eval()
    return model, args, cfg, predictor


def get_data(name="coco_2017_val"):
    """
    A warp function to get the dataset and metadata from D2.

    Args:
    -- name: dataset name.

    Returns:
    -- dataset.
    -- metadata.
    """
    try:
        if 'kitti' not in name:
            dataset = detectron2.data.get_detection_dataset_dicts((name,))
            metadata = MetadataCatalog.get(name)
        else:
            if 'semantics' in name:
                try:
                    dataset_instance_name = "kitti_semantics_instance_train"
                    dataset = detectron2.data.get_detection_dataset_dicts((dataset_instance_name,))
                    metadata = MetadataCatalog.get(dataset_instance_name)
                except:
                    # do the setup for KITTI
                    from detectron2.data import DatasetCatalog, MetadataCatalog
                    from detectron2.data.datasets import builtin_meta
                    from detectron2.data.datasets.cityscapes import load_cityscapes_instances

                    meta = builtin_meta._get_builtin_metadata("cityscapes")

                    dataset_instance_name = "kitti_semantics_instance_train"
                    image_split_dir = "/home/devenish/Desktop/delight/datasets/kitti_semantics_cs/data_semantics/train"
                    gt_split_dir = "/home/devenish/Desktop/delight/datasets/kitti_semantics_cs/gtFine/train"

                    # from_json = True if ground truth json annotation file is available
                    DatasetCatalog.register(dataset_instance_name,
                                            lambda x=image_split_dir, y=gt_split_dir: load_cityscapes_instances(
                                                x, y, from_json=False, to_polygons=True))
                    MetadataCatalog.get(dataset_instance_name).set(image_dir=image_split_dir, gt_dir=gt_split_dir,
                                                                evaluator_type="cityscapes_instance", **meta)
                    print("finish registering {} to DatasetCatalog.".format(dataset_instance_name))

                    dataset = detectron2.data.get_detection_dataset_dicts((dataset_instance_name,))
                    metadata = MetadataCatalog.get(dataset_instance_name)
            elif 'detection' in name:
                dataset_instance_name = "kitti_detection_2d"
                path = "/home/devenish/Desktop/delight/datasets/kitti_object/"
                # try:
                #     dataset = detectron2.data.get_detection_dataset_dicts((dataset_instance_name,))
                #     metadata = MetadataCatalog.get(dataset_instance_name)
                # except:
                # if dataset_instance_name in DatasetCatalog.list():
                #     DatasetCatalog.remove(dataset_instance_name)

                # do the setup for KITTI
                from detectron2.data import DatasetCatalog, MetadataCatalog
                from detectron2.data.datasets import builtin_meta
                from detectron2.data.datasets.cityscapes import load_cityscapes_instances

                print(DatasetCatalog.list())
                if dataset_instance_name in DatasetCatalog.list():
                    DatasetCatalog.remove(dataset_instance_name)

                meta = builtin_meta._get_builtin_metadata("cityscapes")

                categories = ['car', 'bicycle', 'pedestrian']
                get_dicts = lambda p=path, c=categories: get_kitti_dicts(path=p, categories=c)

                DatasetCatalog.register(dataset_instance_name, get_dicts)
                MetadataCatalog.get(dataset_instance_name)
                print("finish registering {} to DatasetCatalog.".format(dataset_instance_name))

                dataset = detectron2.data.get_detection_dataset_dicts((dataset_instance_name,))
                metadata = MetadataCatalog.get(dataset_instance_name)

            else:
                raise NotImplementedError

    except BaseException:
        raise NotImplementedError("no such dataset")
    return dataset, metadata
