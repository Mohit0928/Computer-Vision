# Computer Vision Demo

This is a computer vision model deployed in Django framework for object detection, Instance Segmentation, and vehicle speed extraction from video and images.
I used Detectron 2 to buld this model.

It is divided in two parts:
1) CV on Images (Object Classification, Instance segmentation, Panoptic Segmentation, Object Detection, and Vehicle Plate Recognizer)
    1) Object Detection
  
    ![](https://ml4a.github.io/images/figures/localization-detection.png)

    2) Instance Segmentation

    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/seg_img.png)

    3) Panoptic Segmentation

    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/panoptic_img.png)

    4) Object Detection

    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/obd_img.png)

    5) Vehicle Plate Recognizer

    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/2uuuk.jpg)
    <br/>
    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/down_arrow.png)
    <br/>
    ![](https://github.com/Mohit0928/Computer-Vision/blob/master/deployment/media/license_img.png)
    
 2) CV on Video (Instance Segmentation, Object Detection)
 
    1) Instance Segmentation
    
    2) Object Detection

## Deployment on local server

cd Computer-Vision/deployment

python manage.py runserver

It has two web app

1) CV for images (cv): It can be deployed at 127.0.0.1:8000/cv/
2) CV for videos (cv_vid): It can be deployed at 127.0.0.1:8000/cv_vid/

## Requirements
You need to install Detectron2. You can see [installation instructions here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

