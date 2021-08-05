from django.shortcuts import render, redirect

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import joblib

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

def base(request):
        
    return render(request, 'cv/base.html')        


def classification(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        # `img` is a PIL image of size 224x224
        img_file_ = settings.BASE_DIR + '/' + img_file
        img = image.load_img(img_file_, target_size=(224, 224))
        # `x` is a float32 Numpy array of shape (224, 224, 3)
        x = image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Finally we preprocess the batch
        # (this does channel-wise color normalization)
        x = preprocess_input(x)
        model = VGG16(weights='imagenet')
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])
        pred = decode_predictions(preds, top=1)[0][0][1]
        #return render(request, 'cv/upload.html', {'uploaded_file_url': uploaded_file_url})
        return render(request, 'cv/classification.html', {'original_img': img_file,
                                                            'prediction': pred})
        
    return render(request, 'cv/classification.html')        

def load_model():
    # load the model for inference 
    model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    return model

def get_segmentation(img_file, model):
    input_image = Image.open(img_file)
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions


label_colors = np.array([(0, 0, 0),  # 0=background
              # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
              (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

def seg2rgb(preds):
    colors = label_colors
    colors = label_colors.astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    rgb = Image.fromarray(preds.byte().cpu().numpy())#.resize(preds.shape)
    rgb.putpalette(colors)
    return rgb


def semantic_segmentation(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        # `img` is a PIL image of size 224x224
        img_file_ = settings.BASE_DIR + '/' + img_file
        img = Image.open(img_file_)
        model = load_model()
        preds = get_segmentation(img_file_, model)
        rgb = seg2rgb(preds)
        
        seg_file = settings.MEDIA_ROOT + '/seg_img.png' 
        rgb.save(seg_file)

        return render(request, 'cv/semantic_segmentation.html', {'original_img': img_file,
                                                                 'segmented_img': '/media/seg_img.png'})
        
    return render(request, 'cv/semantic_segmentation.html') 

def panoptic_segmentation(request):
    if request.method=='POST' and request.FILES["myfile"]:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        img_file_ = settings.BASE_DIR + '/' + img_file
        im=cv2.imread(img_file_)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        predictor = DefaultPredictor(cfg)
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

        panoptic_file = settings.MEDIA_ROOT + '/panoptic_img.png' 
        cv2.imwrite(panoptic_file, out.get_image()[:, :, ::-1])

        return render(request, 'cv/panoptic_segmentation.html', {'original_img': img_file,
                                                            'panoptic_img': '/media/panoptic_img.png'})


    return render(request,'cv/panoptic_segmentation.html')


def object_detection(request):
    if request.method=='POST' and request.FILES["myfile"]:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        img_file_ = settings.BASE_DIR + '/' + img_file
        im=cv2.imread(img_file_)

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        obd_file = settings.MEDIA_ROOT + '/obd_img.png' 
        cv2.imwrite(obd_file, out.get_image()[:, :, ::-1])

        return render(request, 'cv/object_detection.html', {'original_img': img_file,
                                                            'obd_img': '/media/obd_img.png'})


    return render(request,'cv/object_detection.html')


def recognize_plate(img, xmin, ymin, xmax, ymax):
   
  box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

  gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
  gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  
  ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
  
  # find contours of regions of interest within license plate
  try:
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  except:
    ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
  im2 = gray.copy()

  return im2


def license_plate(request):
    if request.method=='POST' and request.FILES["myfile"]:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        img_file = fs.url(filename)
        
        img_file_ = settings.BASE_DIR + '/' + img_file
        im=cv2.imread(img_file_)
        print(im)

        predictor=joblib.load('license_plate_recognizer.sav')
        
        outputs = predictor(im)
        coords=outputs["instances"].pred_boxes.tensor.cpu().numpy()[0]
        xmin, ymin, xmax, ymax = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
        img=recognize_plate(im, xmin, ymin, xmax, ymax)
        print(img)
        license_file = settings.MEDIA_ROOT + '/license_img.png' 
        cv2.imwrite(license_file, img)

        return render(request, 'cv/license_plate.html', {'original_img': license_file,
                                                            'license_img': '/media/license_img.png'})

    return render(request,'cv/license_plate.html')


