import codecs
from os import path
from setuptools import setup


setup(
    name='yolov7face',
    version='0.0.6',
    description='Python library for face detection and anonymization based on YOLOv7 models.',
    keywords=['python', 'yolo', 'yolov7', 'face', 'face detection', 'anonymization', 'computer vision'],
    author='Mehdi Samsami',
    author_email='mehdisamsami@live.com',
    url='https://github.com/msamsami/yolov7face',
    long_description=codecs.open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=['yolov7face'],
    classifiers=[
        'Topic :: Computer Vision',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires=">=3.7",
    install_requires=[
        'matplotlib>=3.2.2', 'numpy>=1.18.5', 'opencv-python>=4.1.1', 'Pillow>=7.1.2', 'PyYAML>=5.3.1',
        'requests>=2.23.0', 'scipy>=1.4.1', 'torch>=1.7.0,!=1.12.0', 'torchvision>=0.8.1,!=0.13.0', 'tqdm>=4.41.0',
        'protobuf<4.21.3', 'tensorboard>=2.4.1', 'pandas>=1.1.4', 'onnx>=1.9.0', 'ipython', 'psutil', 'thop'
    ],
    extras_require={}
)
