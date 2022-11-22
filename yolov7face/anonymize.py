from typing import Tuple, Optional, List
from typing_extensions import TypedDict

import cv2
import numpy as np
from PIL import Image
import torch


__all__ = ['FaceAnonymizer']


class _AnonymizerParams(TypedDict):
    method: str
    bbox_type: str
    blur_kernel_size: Optional[Tuple[int, int]]
    blur_sigma_x: Optional[float]
    blur_sigma_y: Optional[float]
    pixelate_type: Optional[str]
    pixelate_size: Optional[Tuple[int, int]]
    pixelate_ratio: Optional[float]
    block_intensity: Optional[int]


class FaceAnonymizer:
    __params_default: _AnonymizerParams = dict(
        method='blur',
        bbox_type='xyxy',
        blur_kernel_size=(11, 11),
        blur_sigma_x=10,
        blur_sigma_y=10,
        pixelate_type='static',
        pixelate_size=(8, 8),
        pixelate_ratio=10,
        block_intensity=0
    )
    __params: _AnonymizerParams = None

    def __init__(self, method: str = 'blur', bbox_type: str = 'xyxy', blur_kernel_size: Optional[Tuple[int, int]] = None,
                 blur_sigma_x: Optional[float] = None, blur_sigma_y: Optional[float] = None,
                 pixelate_type: Optional[str] = None, pixelate_size: Optional[Tuple[int, int]] = None,
                 pixelate_ratio: Optional[float] = None, block_intensity: Optional[int] = None):
        """Initializes an instance of the class to anonymize faces in the image according to given parameters.

        Args:
            method (str): Anonymization method; either 'blur', 'pixelate', or 'block'. Defaults to 'blur'.
            bbox_type (str): Type of bounding box values, either 'xyxy' or 'xywh'. Defaults to 'xyxy'.
            blur_kernel_size (Optional[Tuple[int, int]]): Gaussian blurring kernel size in the form of (height, width).
                                                          Defaults to None for (11, 11).
            blur_sigma_x (Optional[float]): Gaussian blurring kernel standard deviation along X-axis (horizontal).
                                            Defaults to None for 10.
            blur_sigma_y (Optional[float]): Gaussian blurring kernel standard deviation along Y-axis (vertical).
                                            Defaults to None for 10.
            pixelate_type (Optional[str]): How to resize image for pixelation; either 'static' to scale down to a fixed
                                           size given by pixelate_size, or 'dynamic' to scale down by a specified ratio
                                           given by pixelate_ratio. Defaults to None for 'static'.
            pixelate_size (Optional[Tuple[int, int]]): Target size in image scale down when pixelate_type is 'static'.
                                                       Defaults to None for (8, 8).
            pixelate_ratio (Optional[float]): Ratio to scale down the image when pixelate_type is 'dynamic'.
                                              Defaults to None for 10.
            block_intensity (Optional[int]): Intensity value (between 0 and 255) by which the face segment will be blocked.
                                             Defaults to None for 0 (black).

        Returns:
            self: The instance itself.
        """
        self.__params = dict(
            method=method,
            bbox_type=bbox_type,
            blur_kernel_size=blur_kernel_size if blur_kernel_size else self.__params_default['blur_kernel_size'],
            blur_sigma_x=blur_sigma_x if blur_sigma_x else self.__params_default['blur_sigma_x'],
            blur_sigma_y=blur_sigma_y if blur_sigma_y else self.__params_default['blur_sigma_y'],
            pixelate_type=pixelate_type if pixelate_type else self.__params_default['pixelate_type'],
            pixelate_size=pixelate_size if pixelate_size else self.__params_default['pixelate_size'],
            pixelate_ratio=pixelate_ratio if pixelate_ratio else self.__params_default['pixelate_ratio'],
            block_intensity=block_intensity if block_intensity else self.__params_default['block_intensity']
        )

    def anonymize(self, img: np.ndarray, faces: List[list], only_faces: bool = False) -> np.ndarray:
        """Performs face anonymization on the input image.

        Args:
            img (np.ndarray): Input image.
            faces (list[list]): Bounding boxes of each face in the image.
            only_faces (bool): If True, only the pixels corresponding with anonymized face segments will be
                               kept on the image while the remaining pixels are 0. Defaults to False.

        Returns:
            np.ndarray: Image after applying anonymization on faces.
        """
        h, w = img.shape[0], img.shape[1]

        face_segments = []
        for bounding_box in faces:
            x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox=bounding_box, img_w=w, img_h=h)
            face_segments.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'face': img[y1:y2, x1:x2, :]})

        new_img = np.zeros(img.shape) if only_faces else img.copy()
        for face_segment in face_segments:
            if self.method == 'blur':
                face = cv2.GaussianBlur(
                    src=face_segment['face'],
                    ksize=self.blur_kernel_size,
                    sigmaX=self.blur_sigma_x,
                    sigmaY=self.blur_sigma_y
                )

            elif self.method == 'pixelate':
                if self.pixelate_type == 'static':
                    target_size = self.pixelate_size
                else:
                    target_size = tuple((np.array(face_segment['face'].shape[:2]) / self.pixelate_ratio).astype(int))
                target_size = tuple([v if v >= 2 else 2 for v in target_size])

                face_small = Image.fromarray(face_segment['face']).resize(target_size, resample=Image.BILINEAR)
                face = np.array(face_small.resize(Image.fromarray(face_segment['face']).size, Image.NEAREST))

            elif self.method == 'block':
                face = face_segment['face']
                face[:, :] = self.block_intensity

            else:
                raise ValueError(
                    f"'{self.method}' is not a valid method; must be either 'blur', 'pixelate', or 'block'")

            x1, y1, x2, y2 = face_segment['x1'], face_segment['y1'], face_segment['x2'], face_segment['y2']
            new_img[y1:y2, x1:x2, :] = face

        return new_img

    def _convert_bbox_to_xyxy(self, bbox: list, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        bbox = bbox if isinstance(bbox[0], (int, float)) else torch.tensor(bbox).int()

        if self.bbox_type == 'xywh':
            bbox = bbox if type(bbox[0]) is int else np.array(bbox) * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = bbox.astype(np.int)
        else:
            x1, y1, x2, y2 = bbox

        return x1, y1, x2, y2

    @property
    def method(self) -> str:
        """Anonymization method; either 'blur', 'pixelate', or 'block'.

        """
        return self.__params.get('method')

    @method.setter
    def method(self, value: str):
        if value not in ['blur', 'pixelate', 'block']:
            raise ValueError(f"'{value}' is not a valid method; must be either 'blur', 'pixelate', or 'block'")
        self.__params['method'] = value if value else self.__params_default['method']

    @property
    def bbox_type(self) -> str:
        """Type of bounding box values, either 'xyxy' or 'xywh'.

        """
        return self.__params.get('bbox_type')

    @bbox_type.setter
    def bbox_type(self, value: str):
        if value not in ['xyxy', 'xywh']:
            raise ValueError(f"'{value}' is not a valid bbox_type; must be either 'xyxy' or 'xywh'")
        self.__params['bbox_type'] = value if value else self.__params_default['bbox_type']

    @property
    def blur_kernel_size(self) -> Optional[Tuple[int, int]]:
        """Gaussian blurring kernel size in the form of (height, width).

        """
        return self.__params.get('blur_kernel_size')

    @blur_kernel_size.setter
    def blur_kernel_size(self, value: Optional[Tuple[int, int]]):
        self.__params['blur_kernel_size'] = value if value else self.__params_default['blur_kernel_size']

    @property
    def blur_sigma_x(self) -> Optional[float]:
        """Gaussian blurring kernel standard deviation along X-axis (horizontal direction).

        """
        return self.__params.get('blur_sigma_x')

    @blur_sigma_x.setter
    def blur_sigma_x(self, value: Optional[float]):
        self.__params['blur_sigma_x'] = value if value else self.__params_default['blur_sigma_x']

    @property
    def blur_sigma_y(self) -> Optional[float]:
        """Gaussian blurring kernel standard deviation along Y-axis (vertical direction).

        """
        return self.__params.get('blur_sigma_y')

    @blur_sigma_y.setter
    def blur_sigma_y(self, value: Optional[float]):
        self.__params['blur_sigma_y'] = value if value else self.__params_default['blur_sigma_y']

    @property
    def pixelate_type(self) -> Optional[str]:
        """How to resize image for pixelation; either 'static' to scale down to a fixed size given by pixelate_size,
         or 'dynamic' to scale down by a specified ratio given by pixelate_ratio.

        """
        return self.__params.get('pixelate_type')

    @pixelate_type.setter
    def pixelate_type(self, value: Optional[str]):
        if value not in ['static', 'dynamic']:
            raise ValueError(f"'{value}' is not a valid pixelate_type; must be either 'static' or 'dynamic'")
        self.__params['pixelate_type'] = value if value else self.__params_default['pixelate_type']

    @property
    def pixelate_size(self) -> Optional[Tuple[int, int]]:
        """Target size in image scale down when pixelate_type is 'static'.

        """
        return self.__params.get('pixelate_size')

    @pixelate_size.setter
    def pixelate_size(self, value: Optional[Tuple[int, int]]):
        self.__params['pixelate_size'] = value if value else self.__params_default['pixelate_size']

    @property
    def pixelate_ratio(self) -> Optional[float]:
        """Ratio to scale down the image when pixelate_type is 'dynamic'.

        """
        return self.__params.get('pixelate_ratio')

    @pixelate_ratio.setter
    def pixelate_ratio(self, value: Optional[float]):
        self.__params['pixelate_ratio'] = value if value else self.__params_default['pixelate_ratio']

    @property
    def block_intensity(self) -> Optional[int]:
        """Intensity value (between 0 and 255) by which the face segment will be blocked.

        """
        return self.__params.get('block_intensity')

    @block_intensity.setter
    def block_intensity(self, value: Optional[int]):
        self.__params['block_intensity'] = value if value else self.__params_default['block_intensity']

    def __repr__(self) -> str:
        return f"<FaceAnonymizer(method='{self.method}', bbox_type='{self.bbox_type}')>"
