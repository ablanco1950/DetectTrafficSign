# DetectTrafficSign
Creation of a model based on yolov8 that uses the file downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data as a custom dataset to detect traffic signs.
The detected signals can be recognized with one of the signal recognition projects published on github.

Requirements

yolo must be installed, if not, follow the instructions indicated in: https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

pip install ultralytics

also must be installed the usual modules in computer vision: numpy, cv2, os, …...

All needed modules can be installed with a plain pip

Functioning:

Download the project to a folder on disk.

The project incorporates the model that is needed: bestDetectTrafficSign.pt:

If you would like to know the procedure followed to obtain it, or execute it by changing parameters, consult the attached document Steps_Creating_ModelYolov8_bestDetectTrafficSign.docx that is attached to the project.

Functioning:

run the program

VIDEODetectTrafficSignalYolov8.py

which uses the traffic-sign-to-test.mp4 video downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data

A more extensive test can be obtained by downloading the video production_id_4434242 (2160p).mp4 from the address https://www.pexels.com/video/road-trip-4434242/ (free download), for which it is enough to activate instruction 16

A recognition test of the detected signs can be performed by downloading the model.h5 model from the project [https://github.com/faeya/traffic-sign-classification](https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition) and running the program:

VIDEORecognizeTrafficSignalYolov8.py

The names of the recognized signals appear on the console.

For step-by-step monitoring of the detected signals, programs are used that collect the detected signals from a set of images in the Test folder.

DetectTrafficSignYolov8.py

Images with the detected signals appear on the console and at the end the total image with the detected images boxed.

Observations:

- In detecting signals, the important thing is to box the detected signals, ignoring the meaning of {0: 'prohibitory', 1: 'danger', 2: 'mandatory', 3: 'other'} since they are not It tries to classify undetected images.
- In the database that has been downloaded to create the model https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data it seems that some signs such as the passage of pedestrians have not been labeled
- I plan to start a project that includes a greater number of images.

References:

https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/code

https://www.kaggle.com/datasets/barzansaeedpour/traffic-sign-detection

https://www.kaggle.com/code/valentynsichkar/traffic-signs-detection-by-yolo-v3-opencv-keras

https://www.pexels.com/video/road-trip-4434242/

https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition

https://github.com/faeya/traffic-sign-classification

https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR




