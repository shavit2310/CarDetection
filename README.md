#YOLO - SSD # Car-detection based traind model

The goal of this project is to write a YOLO object detector use to detect objects in both images and video streams using Deep Learning, OpenCV, ffmpeg.
## NOTE: due to too-big weights file, the weights is not in the repo. please download it from https://pjreddie.com/media/files/yolov3.weights  !!!
## implementaition stages
  We’ll be using YOLOv3, in particular, YOLO trained on the COCO dataset. 
  In order to use a trained model, we will use the weight and config outputs. 
  
  These are our implementaion steps:
    * Use YOLO to detect objects.
    * Select only detected objects having confidence higher than a set threshold.
    * Only select cars in the remaining detected objects.
    * Run the pipeline on images.
    * Save items box detection in a json file.
    * Run the pipeline on a video stream.
  
## Dependencies
For that project, I updated my environment, and I'm using Keras 2.4.3.
ffmpeg-python (not complited yet)  0.2.0

## YOLO Theory
YOLO (or You Only Look Once) is an object detection pipeline based on a Neural Network. It uses the following approach. A single neural network is applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

This model has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation.

YOLOv2 uses a few tricks to improve training and increase performance. Like Overfeat and SSD, it uses a fully-convolutional model, but it is still trained on whole images, not hard negatives. It adjust priors on bounding boxes instead of predicting the width and height outright. However, it still predicts the x and y coordinates directly.

Source : Matthijs Hollemans

YOLO divides up the image into a grid. Each of the cells is responsible for predicting bounding boxes. A bounding box describes the rectangle that encloses an object.

Confidence score: YOLO outputs a confidence score that tells us how certain it is that the predicted bounding box actually encloses some object. This score doesn’t say anything about what kind of object is in the box, just if the shape of the box is any good. It's the Confidence rate.

Lable: For each bounding box, the cell also predicts a class. Like a classifier, it gives a probability distribution over all the possible classes. 
It can detect up to 20 different classes such as: bicycle, boat, car, cat, dog, person,…

The confidence score for the bounding box and the class prediction are combined into one final score that tells us the probability that this bounding box contains a specific type of object. For example, the big fat yellow box on the left is 85% sure it contains the object “dog”


## Implementation
Based on this site: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
The tree directories is as followed:
4 directories: main folder for the configurations of the trained model (3 files): yolo-coco, two for the input media: images, videos and one for the output videos: output.
6 files: 3 in yolo-coco directory - coco.names, yolov3.cfg, yolov3.weights. 3 in the project level: main.py, CV2_for_video.py, ffmpeg_for_video.py

To run the script from terminal:
  * the file extenstion are defined (can be updated) for image: gif, png or jpg. output-txt. for video: input-mp4 or gif. output-avi.
For Video- python main.py --input videos/name_of_video.mp4 --output output/same_name_of_input_file.avi --yolo yolo-coco

For Image- python main.py --input images/name_of_image.gif --output output/same_name_of_input_file.txt --yolo yolo-coco


## References
  * Yolo/Darknet website: https://pjreddie.com/darknet/yolo/
  * https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
  * Object detection with YOLO (YOLO explanation)
  *  https://github.com/shavit2310/CarND-Vehicle-Detection/blob/master/README.md

# Still todo
implement ffmpeg for video
