import os
from typing import Optional
import warnings

import requests


class YOLOv7Model:
    """This class represents a YOLOv7 model.

    """
    _issue_warning = True

    def __init__(self, name: str, filepath: str, url: Optional[str] = None, version: Optional[str] = None,
                 author: Optional[str] = None):
        """Creates an instance of the class with given parameters.

        Args:
            name (str): Name of the model.
            filepath (str): Filepath to model weights (i.e., *.pt file).
            url (Optional[str]): URL of the model weights, in case *.pt file is not downloaded in filepath.
                                 Defaults to None.
            version (Optional[str]): Model version. Defaults to None.
            author (Optional[str]): Model author. Defaults to None.

        """
        self._name = name
        self._filepath = filepath
        self._url = url
        self._version = version
        self._author = author

    def _check_filepath_and_url(self):
        if not os.path.isfile(self._filepath) and self._url is None:
            raise ValueError(
                "Model does not exist in filepath. Please provide a valid filepath, or a valid url in case the model "
                "should be downloaded to filepath"
            )
        elif not os.path.isfile(self._filepath) and self._url is not None:
            print("Downloading model weights from url...")
            resp = requests.get(self._url)
            open(self._filepath, 'wb').write(resp.content)
            print("Model weights downloaded and saved to filepath")
        elif os.path.isfile(self._filepath) and self._url is not None:
            if self._issue_warning:
                warnings.warn("Model already exists in filepath; url will be ignored")
                self._issue_warning = False
        else:
            pass

    @property
    def name(self) -> str:
        """Name of the model.

        """
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def filepath(self) -> str:
        """Filepath to model weights (i.e., *.pt file).

        """
        self._check_filepath_and_url()
        return self._filepath

    @filepath.setter
    def filepath(self, value: str):
        self._filepath = value

    @property
    def url(self) -> Optional[str]:
        """URL of the model weights, in case *.pt file is not downloaded in filepath.

        """
        return self._url

    @url.setter
    def url(self, value: Optional[str]):
        self._url = value

    @property
    def version(self) -> Optional[str]:
        """Model version.

        """
        return self._version

    @version.setter
    def version(self, value: Optional[str]):
        self._version = value

    @property
    def author(self) -> Optional[str]:
        """Model author.

        """
        return self._author

    @author.setter
    def author(self, value: Optional[str]):
        self._author = value

    def __repr__(self) -> str:
        return (
            f"<YOLOv7Model(name='{self.name}', filepath='{os.path.basename(self._filepath)}'"
            f", version={self.version}" if self.version else ''
            f", author='{self.author}'" if self.author else ''
            ")>"
        )


DEFAULT_MODEL = YOLOv7Model(
    name="YOLOv7-WIDERFACE",
    filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov7-widerface.pt"),
    url="https://www.dropbox.com/s/l6rx9oxgihjhc88/yolov7-widerface.pt?dl=1",
    version="0.0.1",
    author="Mehdi Samsami"
)
