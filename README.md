# Product-Recognition-with-YOLOv3
Product recognition on Store Shelves with YoloV3


The goal of this project is to use computer vision techniques to perform object detection. In particular, we want a system able to recognize products on store shelves.
The simple solution proposed here combines both state of the art detectors based on deep learning and traditional techniques for object detection.

The network used in this project is YOLOv3, which is available at https://github.com/AlexeyAB/darknet. YOLO has been used for finding the bounding boxes of the product in shelves. Each region is then cropped and classified using RootSIFT, an improved versione of SIFT.

The mAP computed on a test set of 70 labeled images of store shelves is 65%.
The gif shows on the left the current scene and on the right the cropped object from scene with the corresponding model prediction.
![](demo.gif)

TaskA is a simple object detector implemented with only SIFT.
TaskB is the object detector used for producing the gif above. For more information please read the report.

Requirements:
numpy, openCV, [pydarknet](https://github.com/madhawav/YOLO3-4-Py)

usage for task b:

python multipleInstancesDetection.py -m models/ -s scenes/
