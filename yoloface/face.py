from typing import Optional
from typing_extensions import TypedDict


class YOLOv7Configs:
    class __params(TypedDict):
        weights: str
        cfg: str
        img_size: int
        conf_thres: float
        iou_thres: float
        device: str
        classes: Optional[list]

    def __init__(self, weights: str, cfg: str, img_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45,
                 device: str = '0', classes: Optional[list] = None):
        """Initializes an instance of the class to hold YOLOv7 model configurations.

        Args:
            weights (str): Path to the weights of the pre-trained model, i.e., 'model.pt' file.
            cfg (str): Path to the YOLOv7 YAML config file used in training.
            img_size (int): Inference image size (in pixels). Defaults to 640.
            conf_thres (float): Object confidence threshold for inference. Defaults to 0.25.
            iou_thres (float): IOU threshold for non-maximum suppression (NMS) for inference. Defaults to 0.45.
            device (str): Device to run the model, e.g., '0', '1', '2', '3', or 'cpu'. Defaults '0'.
            classes (Optional[list]): List of classes to filter. Defaults to None.

        Returns:
            self: The instance itself.
        """
        self.__params = dict(
            weights=weights,
            cfg=cfg,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
            classes=classes
        )

    @property
    def weights(self) -> str:
        """Path to the weights of the pre-trained model (i.e., 'model.pt' file).

        """
        return self.__params.get('weights')

    @weights.setter
    def weights(self, value: str):
        self.__params['weights'] = value

    @property
    def cfg(self) -> str:
        """Path to the YOLOv7 YAML config file used in training.

        """
        return self.__params.get('cfg')

    @cfg.setter
    def cfg(self, value: str):
        self.__params['cfg'] = value

    @property
    def img_size(self) -> int:
        """Inference image size (in pixels).

        """
        return self.__params.get('img_size')

    @img_size.setter
    def img_size(self, value: int):
        self.__params['img_size'] = value

    @property
    def conf_thres(self) -> float:
        """Object confidence threshold for inference.

        """
        return self.__params.get('conf_thres')

    @conf_thres.setter
    def conf_thres(self, value: float):
        self.__params['conf_thres'] = value

    @property
    def iou_thres(self) -> float:
        """IOU threshold for non-maximum suppression (NMS) for inference.

        """
        return self.__params.get('iou_thres')

    @iou_thres.setter
    def iou_thres(self, value: float):
        self.__params['iou_thres'] = value

    @property
    def device(self) -> str:
        """Device to run the model, e.g., '0', '1', '2', '3', or 'cpu'.

        """
        return self.__params.get('device')

    @device.setter
    def device(self, value: str):
        self.__params['device'] = value

    @property
    def classes(self) -> Optional[list]:
        """List of classes to filter.

        """
        return self.__params.get('classes')

    @classes.setter
    def classes(self, value: Optional[list]):
        self.__params['classes'] = value

    def __repr__(self) -> str:
        return (
            f"<YOLOv7Configs(weights='{self.weights}', cfg='{self.cfg}', img_size={self.img_size}, "
            f"conf_thres={self.conf_thres}, iou_thres={self.iou_thres}, device='{self.device}')>"
        )
