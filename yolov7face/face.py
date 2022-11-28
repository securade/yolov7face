from base64 import b64encode, b64decode
import io
import os
from typing import Union, Optional, List, Tuple
from typing_extensions import TypedDict
import warnings

import cv2
from IPython.display import display, Javascript
from moviepy.editor import AudioFileClip, VideoClip
import numpy as np
from numpy import random
import PIL
import torch
import yaml

from .anonymize import FaceAnonymizer
from ._colab import eval_js, cv2_imshow
from .model import YOLOv7Model
from .utils import letterbox

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized


__all__ = ['YOLOv7Configs', 'YOLOv7Face']


class _YOLOv7ConfigsParams(TypedDict):
    weights: Union[str, YOLOv7Model]
    cfg: Optional[dict]
    img_size: int
    conf_thres: float
    iou_thres: float
    device: str
    classes: Optional[list]


class YOLOv7Configs:
    __params: _YOLOv7ConfigsParams = None

    def __init__(self, weights: Union[str, YOLOv7Model], cfg: Optional[Union[str, dict]] = None, img_size: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = 'cpu', classes: Optional[list] = None):
        """Initializes an instance of the class to hold YOLOv7 model configurations.

        Args:
            weights (Union[str, YOLOv7Model]): Path to the weights of the pre-trained model (i.e., *.pt file), or an
                                               instance of YOLOv7Model class with valid filepath.
            cfg (Optional[Union[str, dict]]): YOLOv7 configs used in training; path to the YAML file or the dictionary
                                              containing the configs. Defaults to None.
            img_size (int): Inference image size (in pixels). Defaults to 640.
            conf_thres (float): Object confidence threshold for inference. Defaults to 0.25.
            iou_thres (float): IOU threshold for non-maximum suppression (NMS) for inference. Defaults to 0.45.
            device (str): Device to run the model, e.g., '0', '1', '2', '3', or 'cpu'. Defaults 'cpu'.
            classes (Optional[list]): List of classes to filter. Defaults to None.

        Returns:
            self: The instance itself.
        """
        cfg_value = cfg
        if cfg and isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg_value = yaml.safe_load(f)

        self.__params = dict(
            weights=weights,
            cfg=cfg_value,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
            classes=classes
        )

    @property
    def weights(self) -> str:
        """Path to the weights of the pre-trained model (i.e., *.pt file).

        """
        weights = self.__params.get('weights')
        return weights if isinstance(weights, str) else weights.filepath

    @weights.setter
    def weights(self, value: Union[str, YOLOv7Model]):
        self.__params['weights'] = value

    @property
    def cfg(self) -> Optional[dict]:
        """YOLOv7 configs used in training.

        """
        return self.__params.get('cfg')

    @cfg.setter
    def cfg(self, value: Union[dict, str]):
        cfg_value = value
        if isinstance(value, str):
            with open(value, "r") as f:
                cfg_value = yaml.safe_load(f)

        self.__params['cfg'] = cfg_value

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
            f"<YOLOv7Configs(weights='{os.path.basename(self.weights)}', img_size={self.img_size}, "
            f"conf_thres={self.conf_thres}, iou_thres={self.iou_thres}, device='{self.device}')>"
        )


class YOLOv7Face:
    def __init__(self, configs: YOLOv7Configs, anonymizer: Optional[FaceAnonymizer] = None, display_n_faces: bool = False,
                 verbose: bool = False):
        """Initializes an instance of the class to use YOLOv7 model for face detection and/or anonymization.

        Args:
            configs (YOLOv7Configs): YOLOv7 model configurations.
            anonymizer (Optional[FaceAnonymizer]): An optional FaceAnonymizer if willing to apply face anonymizer.
                                                   Defaults to None.
            display_n_faces (bool): Whether to display the number of detected faces on the output image.
                                    Defaults to False.
            verbose (bool): Verbosity level. Defaults to False.

        Returns:
            self: The instance itself.
        """
        self.configs: YOLOv7Configs = configs
        self.anonymizer: Optional[FaceAnonymizer] = anonymizer
        self.display_n_faces: bool = display_n_faces
        self.verbose = verbose

        if self.anonymizer and self.anonymizer.bbox_type != 'xyxy':
            self.anonymizer.bbox_type = 'xyxy'
            warnings.warn("anonymizer.bbox_type must be 'xyxy'. It was automatically changed to 'xyxy'")

    @staticmethod
    def _put_n_faces(img, n_faces: int):
        return cv2.putText(img, text=f"{n_faces or 'no'} face{'s' * (n_faces == 0 or n_faces > 1)}",
                           org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3, color=(255, 0, 0),
                           thickness=2, lineType=cv2.LINE_AA)

    def _initialize_assets(self, gpu_initialization: bool = True):
        set_logging(-1 if self.verbose else 1)
        device = select_device(self.configs.device)
        half = device.type != 'cpu'
        model = attempt_load(self.configs.weights, map_location=device)  # Load FP32 model
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        stride = int(model.stride.max())  # Model stride
        img_size = check_img_size(self.configs.img_size, s=stride)  # Check img_size

        if gpu_initialization and device.type != 'cpu':
            model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

        return model, device, img_size, stride, names, colors, half

    @staticmethod
    def _prepare_img(img0, img_size, stride, device, half):
        img = letterbox(img0, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # Convert [0, 255] to [0.0, 1.0]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def _selected_classes(self, names):
        classes = None
        if self.configs.classes:
            classes = []
            for class_name in self.configs.classes:
                classes.append(names.index(class_name))

        if classes:
            classes = [i for i in range(len(names)) if i not in classes]

        return classes

    def predict_img(self, img: Union[str, PIL.Image.Image, np.ndarray], custom_bbox: Optional[List[list]] = None,
                    custom_bbox_label: Optional[str] = None, view_img: bool = True, save_to: Optional[str] = None,
                    return_inf_time: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Performs face detection (and anonymization if self.anonymizer is provided) on the input image.

        Args:
            img (Union[str, PIL.Image.Image, np.ndarray]): Input image for face detection and/or anonymization.
            custom_bbox (Optional[List[list]]): Custom, optional bounding boxes to display on the output image.
                                                Defaults to None.
            custom_bbox_label (Optional[str]): Optional label to show on custom bounding boxes. Defaults to None.
            view_img (bool): Whether to display the output image. Defaults to True.
            save_to (Optional[str]): If provided, the output image will be saved to this path. Defaults to None.
            return_inf_time (bool): Whether to return the model inference time. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, float]]: Output image after face detection and/or anonymization if
                                                         return_inf_time=False; otherwise, a tuple containing:
                                                            np.ndarray: Output image.
                                                            float: Model inference time.
        """
        if isinstance(img, str):
            img0 = cv2.imread(img)
        elif isinstance(img, PIL.Image.Image):
            img0 = np.array(img)
        else:
            img0 = img.copy()

        with torch.no_grad():

            # Initialize model and required assets
            model, device, img_size, stride, names, colors, half = self._initialize_assets()

            # Prepare image for inference
            img = self._prepare_img(img0, img_size, stride, device, half)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Filter out undesired classes
            classes = self._selected_classes(names)

            # Apply NMS
            pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=classes,
                                       agnostic=False)
            t2 = time_synchronized()

            n_faces = 0
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # Print string
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

                    n_faces = (det[:, -1] == c).sum()

            if self.display_n_faces:
                img0 = self._put_n_faces(img0, n_faces)

            if custom_bbox:
                for bbox in custom_bbox:
                    plot_one_box(bbox, img0, label=custom_bbox_label, line_thickness=2)

        if save_to:
            PIL.Image.fromarray(img0).save(save_to)

        if view_img:
            cv2_imshow(img0)

        if return_inf_time:
            return img0, t2-t1
        return img0

    def predict_video(self, video_path: str, save_to: str, keep_audio: bool = True):
        """Performs face detection (and anonymization if self.anonymizer is provided) on the input video.

        Args:
            video_path (str): Path to the input video for face detection and/or anonymization.
            save_to (str): Path to save the output video.
            keep_audio (str): Whether to keep audio on the input video file. Defaults to True.
        """
        video = cv2.VideoCapture(video_path)

        # Video information
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initializing object for writing video output
        if not keep_audio:
            output = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        else:
            output = []

        torch.cuda.empty_cache()

        with torch.no_grad():

            # Initialize model and required assets
            model, device, img_size, stride, names, colors, half = self._initialize_assets()

            # Filter out undesired classes
            classes = self._selected_classes(names)

            for j in range(n_frames):
                ret, img0 = video.read()

                if ret:
                    # Prepare image for inference
                    img = self._prepare_img(img0, img_size, stride, device, half)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=classes,
                                               agnostic=False)
                    t2 = time_synchronized()

                    n_faces = 0
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
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

                            n_faces = (det[:, -1] == c).sum()

                    if self.display_n_faces:
                        img0 = self._put_n_faces(img0, n_faces)

                    if self.verbose:
                        print(f"{j + 1}/{n_frames} frames processed (inference time={t2-t1:.4f}sec)")

                    if not keep_audio:
                        output.write(img0)
                    else:
                        output.append(img0[:, :, ::-1])
                else:
                    break

        if self.verbose:
            print("Saving the output video...")

        if not keep_audio:
            output.release()
        else:
            def img_frame_at_t(t):
                t_range = np.arange(0, n_frames/fps, 1/fps).tolist()
                idx = t_range.index(t)
                return output[idx]

            audio_clip = AudioFileClip(video_path)
            output_clip = VideoClip(make_frame=img_frame_at_t, duration=n_frames/fps)
            output_clip = output_clip.set_audio(audio_clip)
            output_clip.write_videofile(save_to, fps=fps, codec='libx264', preset='veryfast', logger=None)
            del output, output_clip, audio_clip

        video.release()

        if self.verbose:
            print("Done!")

    @staticmethod
    def js_to_image(js_object) -> np.ndarray:
        """Converts a given JavaScript object into an OpenCV image.

        Args:
            js_object: JavaScript object.

        Returns:
            np.ndarray: OpenCV BGR image.
        """
        # Decode base64 image
        image_bytes = b64decode(js_object.split(',')[1])

        # Convert bytes to numpy array
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode numpy array into OpenCV BGR image
        img = cv2.imdecode(jpg_as_np, flags=1)

        return img

    @staticmethod
    def bbox_to_bytes(bbox_array) -> str:
        """Converts OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream.

        Args:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.

        Returns:
            bytes: Base64 image byte string.
        """
        # Convert array into PIL image
        bbox_pil = PIL.Image.fromarray(bbox_array, 'RGBA')  # RGBA
        io_buffer = io.BytesIO()

        # Format bbox into png for return
        bbox_pil.save(io_buffer, format='png')

        # Format return string
        bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(io_buffer.getvalue()), 'utf-8')))

        return bbox_bytes

    @staticmethod
    def video_stream() -> None:
        """JavaScript to properly create live video stream using webcam as input.

        """
        with open("yolov7face/video_stream.js", "r") as f:
            js_code = ''.join(f.readlines())

        js = Javascript(js_code)
        display(js)

    @staticmethod
    def video_frame(label: str, bbox):
        data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
        return data

    def predict_webcam(self):
        # Start streaming video from webcam
        self.video_stream()
        label_html = 'Capturing...'

        bbox = ''
        with torch.no_grad():

            # Initialize model and required assets
            model, device, _, stride, names, colors, half = self._initialize_assets(False)

            # Set streaming image size
            img_size = (480, 640)

            if device.type != 'cpu':
                model(torch.zeros(1, 3, img_size[0], img_size[1]).to(device).type_as(next(model.parameters())))

            # Filter out undesired classes
            classes = self._selected_classes(names)

            while True:
                js_reply = self.video_frame(label_html, bbox)
                if not js_reply:
                    break

                img0 = self.js_to_image(js_reply["img"])
                bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

                # Prepare image for inference
                img = self._prepare_img(img0, img_size, stride, device, half)

                # Inference
                pred = model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=classes,
                                           agnostic=False)

                n_faces = 0
                for i, det in enumerate(pred):
                    s = ''
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            if self.anonymizer:
                                bbox_array[:, :, :3] = self.anonymizer.anonymize(img=img0, faces=[xyxy],
                                                                                 only_faces=True)
                            else:
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, bbox_array, label=label, color=colors[int(cls)], line_thickness=3)

                        n_faces = (det[:, -1] == c).sum()

                if self.display_n_faces:
                    bbox_array = self._put_n_faces(bbox_array, n_faces)

                bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
                bbox_bytes = self.bbox_to_bytes(bbox_array)

                bbox = bbox_bytes
