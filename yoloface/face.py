from typing import Union, Optional
from typing_extensions import TypedDict
import warnings

import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from numpy import random
from PIL import Image
import torch

from anonymize import FaceAnonymizer
from _utils import letterbox

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized


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


class YOLOv7Face:
    def __init__(self, configs: YOLOv7Configs, anonymizer: Optional[FaceAnonymizer] = None,
                 display_num_faces: bool = True,
                 verbose: bool = False):
        """Initializes an instance of the class to use YOLOv7 model for face detection and/or anonymization.

        Args:
            configs (YOLOv7Configs): YOLOv7 model configurations.
            anonymizer (Optional[FaceAnonymizer]): An optional FaceAnonymizer if willing to apply face anonymizer.
                                                   Defaults to None.
            display_num_faces (bool): Whether to display the number of detected faces in the image. Defaults to True.
            verbose (bool): Verbose level. Defaults to False.

        Returns:
            self: The instance itself.
        """
        self.configs: YOLOv7Configs = configs
        self.anonymizer: Optional[FaceAnonymizer] = anonymizer
        self.display_num_faces: bool = display_num_faces
        self.verbose = verbose

        if self.anonymizer.bbox_type != 'xyxy':
            self.anonymizer.bbox_type = 'xyxy'
            warnings.warn("anonymizer.bbox_type must be set to 'xyxy'. It was automatically changed.")

    def predict_img(self, img: Union[str, Image.Image, np.ndarray], view_img: bool = True,
                    save_to: Optional[str] = None):
        """Performs face detection (and anonymization if self.anonymizer is provided) on the input image.

        Args:
            img (Union[str, Image.Image, np.ndarray]): Input image for face detection and/or anonymization.
            view_img (bool): Whether to display the output image. Defaults to True.
            save_to (Optional[str]): If provided, the output image will be saved to this path. Defaults to None.

        Returns:
            np.ndarray: Output image after face detection and/or anonymization.
        """
        if isinstance(img, str):
            img0 = cv2.imread(img)
        elif isinstance(img, Image.Image):
            img0 = np.array(img)
        else:
            img0 = img.copy()

        with torch.no_grad():
            set_logging(-1 if self.verbose else 1)
            device = select_device(self.configs.device)
            half = device.type != 'cpu'
            model = attempt_load(self.configs.weights, map_location=device)  # Load FP32 model
            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            stride = int(model.stride.max())  # Model stride
            imgsz = check_img_size(self.configs.img_size, s=stride)  # Check img_size

            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # Convert [0, 255] to [0.0, 1.0]
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            classes = None
            if self.configs.classes:
                classes = []
                for class_name in self.configs.classes:
                    classes.append(names.index(class_name))

            if classes:
                classes = [i for i in range(len(names)) if i not in classes]

            pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=classes,
                                       agnostic=False)

            t2 = time_synchronized()
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # Print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # Detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add to string

                    for *xyxy, conf, cls in reversed(det):
                        if self.anonymizer:
                            img0 = self.anonymizer.anonymize(img=img0, faces=[xyxy])
                        else:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

        if save_to:
            Image.fromarray(img0).save(save_to)

        if view_img:
            cv2_imshow(img0)

        return img0

    def predict_video(self, video_path: str, save_to: str):
        """Performs face detection (and anonymization if self.anonymizer is provided) on the input video.

        Args:
            video_path (str): Path to the input video for face detection and/or anonymization.
            save_to (str): Path to save the output video.

        """
        video = cv2.VideoCapture(video_path)

        # Video information
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initializing object for writing video output
        output = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        torch.cuda.empty_cache()

        # Initializing model and setting it for inference
        with torch.no_grad():
            set_logging(-1 if self.verbose else 1)
            device = select_device(self.configs.device)
            half = device.type != 'cpu'
            model = attempt_load(self.configs.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(self.configs.img_size, s=stride)  # check img_size
            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

            classes = None
            if self.configs.classes:
                classes = []
                for class_name in self.configs.classes:
                    classes.append(names.index(class_name))

            if classes:
                classes = [i for i in range(len(names)) if i not in classes]

            for j in range(nframes):
                ret, img0 = video.read()

                if ret:
                    img = letterbox(img0, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # Convert [0, 255] to [0.0, 1.0]
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=classes,
                                               agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            for *xyxy, conf, cls in reversed(det):
                                if self.anonymizer:
                                    img0 = self.anonymizer.anonymize(img=img0, faces=[xyxy])
                                else:
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    if self.verbose:
                        print(f"{j + 1}/{nframes} frames processed")

                    output.write(img0)
                else:
                    break

        output.release()
        video.release()
