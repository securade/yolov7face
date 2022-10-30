from typing import Tuple, Optional, List

import cv2
from PIL import Image
import numpy as np


class FaceAnonymizer:
    __params_default = dict(
        blur_kernel_size=(11, 11),
        blur_sigma_x=10,
        blur_sigma_y=10,
        pixelate_type='static',
        pixelate_size=(8, 8),
        pixelate_ratio=10,
        block_intensity=0
    )

    def __init__(self, method: str = 'blur', bbox_type: str = 'xywh', blur_kernel_size: Optional[Tuple[int, int]] = None,
                 blur_sigma_x: Optional[float] = None, blur_sigma_y: Optional[float] = None,
                 pixelate_type: Optional[str] = 'static', pixelate_size: Optional[Tuple[int, int]] = None,
                 pixelate_ratio: Optional[float] = None, block_intensity: Optional[int] = None):
        """Initializes an instance of the class to anonymize faces in the image according to given parameters.

        Args:
            method (str): Anonymization method; either 'blur', 'pixelate', or 'block'. Defaults to 'blur'.
            bbox_type (str): Type of bounding box values, either 'xyxy' or 'xywh'. Defaults to 'xywh'.
            blur_kernel_size (Optional[Tuple[int, int]]): Gaussian blurring kernel size in the form of (height, width).
                                                          Defaults to None for (11, 11).
            blur_sigma_x (Optional[float]): Gaussian blurring kernel standard deviation along X-axis (horizontal direction).
                                            Defaults to None for 10.
            blur_sigma_y (Optional[float]): Gaussian blurring kernel standard deviation along Y-axis (vertical direction).
                                            Defaults to None for 10.
            pixelate_type (Optional[str]): How to resize image for pixelation; either 'static' to downsample to a fixed size
                                           specified by pixelate_size, or 'dynamic' to downsample by a specified ratio given
                                           by pixelate_ratio. Defaults to None for 'static'.
            pixelate_size (Optional[Tuple[int, int]]): Target size in image downsampling when pixelate_type is 'static'.
                                                       Defaults to None for (8, 8).
            pixelate_ratio (Optional[float]): Ratio to downsample the image when pixelate_type is 'dynamic'. Defaults to
                                              None for 10.
            block_intensity (Optional[int]): Intensity value by which the face segment will be blocked.
                                             Defaults to None for 0 (black).

        Returns:
            self: The instance itself.
        """
        self.method = method
        self.bbox_type = bbox_type
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size else self.__params_default['blur_kernel_size']
        self.blur_sigma_x = blur_sigma_x if blur_sigma_x else self.__params_default['blur_sigma_x']
        self.blur_sigma_y = blur_sigma_y if blur_sigma_y else self.__params_default['blur_sigma_y']
        self.pixelate_type = pixelate_type if pixelate_type else self.__params_default['pixelate_type']
        self.pixelate_size = pixelate_size if pixelate_size else self.__params_default['pixelate_size']
        self.pixelate_ratio = pixelate_ratio if pixelate_ratio else self.__params_default['pixelate_ratio']
        self.block_intensity = block_intensity if block_intensity else self.__params_default['block_intensity']

    def anonymize(self, img: np.ndarray, faces: List[list]) -> np.ndarray:
        """Performs face anonymization on the input image.

        Args:
            img (np.ndarray): Input image.
            faces (list[list]): Bounding boxes of each face in the image.

        Returns:
            np.ndarray: Image after applying anonymization on faces.
        """
        h, w, _ = img.shape

        face_segments = []
        for bounding_box in faces:
            x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox=bounding_box, img_w=w, img_h=h)
            face_segments.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'face': img[y1:y2, x1:x2, :]})

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
                raise ValueError(f"'{self.method}' is not a valid method; must be either 'blur', 'pixelate', or 'block'")

            x1, y1, x2, y2 = face_segment['x1'], face_segment['y1'], face_segment['x2'], face_segment['y2']
            img[y1:y2, x1:x2, :] = face

        return img

    def _convert_bbox_to_xyxy(self, bbox: list, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        bbox = bbox if isinstance(bbox[0], (int, float)) else torch.tensor(bbox).int()

        if self.bbox_type == 'xywh':
            bbox = bbox if type(bbox[0]) is int else np.array(bbox) * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = bbox.astype(np.int)
        else:
            x1, y1, x2, y2 = bbox

        return x1, y1, x2, y2

    def __repr__(self) -> str:
        return f"<FaceAnonymizer(method='{self.method}', bbox_type='{self.bbox_type}')>"
