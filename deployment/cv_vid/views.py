from django.shortcuts import render, redirect
from django.http import HttpResponse,StreamingHttpResponse,HttpResponseServerError

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings


from moviepy.editor import VideoFileClip

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random,time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("coco_2017_train")
coco = metadata.get("thing_classes", None)


def base(request):
        
    return render(request, 'cv_vid/base.html')        
     



# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def get_segmentation(im):
    
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img=out.get_image()[:, :, ::-1]

    return img


def semantic_segmentation(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        video_file = fs.url(filename)

        video_file_ = settings.BASE_DIR + '/' + video_file
            
        
        video_output = 'output.mp4'

        clip = VideoFileClip(video_file_)
        
        def process_frame(frame):        
            

            return get_segmentation(frame)
        
        video_output = settings.MEDIA_ROOT + '/seg_vid.mp4'
        seg_clip = clip.fl_image(process_frame)
        seg_clip.write_videofile(video_output, audio=False)


        return render(request, 'cv_vid/semantic_segmentation.html', {'original_vid': video_file,
                                                                     'segmented_vid': '/media/seg_vid.mp4'})
    return render(request, 'cv_vid/semantic_segmentation.html')   



def get_objects(frame):
    outputs = predictor(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    coords=outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes=outputs["instances"].pred_classes.cpu().numpy()
    scores=outputs["instances"].scores.cpu().numpy()*100

    cars= [coords[:,0],coords[:,1],coords[:,2]-coords[:,0],coords[:,3]-coords[:,1]]
    for i in range(len(cars[0][:])):
        x,y,w,h=int(cars[0][i]),int(cars[1][i]),int(cars[2][i]),int(cars[3][i])
        frame=cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255, 0),3)
        frame=cv2.putText(frame,str(coco[classes[i]])+" "+str(int(scores[i])),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    return frame


def object_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        video_file = fs.url(filename)

        video_file_ = settings.BASE_DIR + '/' + video_file
            
        
        video_output = 'output.mp4'

        clip = VideoFileClip(video_file_)
        
        def process_frame(frame):        
            

            return get_objects(frame)
        
        video_output = settings.MEDIA_ROOT + '/obd_vid.mp4'
        obd_clip = clip.fl_image(process_frame)
        obd_clip.write_videofile(video_output, audio=False)


        return render(request, 'cv_vid/object_detection.html', {'original_vid': video_file,
                                                                     'segmented_vid': '/media/obd_vid.mp4'})
    return render(request, 'cv_vid/object_detection.html')   
