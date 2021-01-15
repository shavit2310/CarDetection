# Yolo object detection with openCV for images and ffmpeg for videos

# From terminal: python CV2_for_video.py --input videos/car_chase_01.mp4 --output output/car_chase_01.avi --yolo yolo-coco
#                python main.py --input imagees/test8.jpg --output output/test8.txt --yolo yolo-coco

# All the issues in one file
# import the necessary packages
import numpy as np
import argparse
import time
import imutils
import json
import cv2
import os

import ffmpg_for_video


def construct_arguments(media):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output video")
    ap.add_argument("-y", "--yolo", required=True,
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.7,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")

    args = vars(ap.parse_args())
    return args


def load_trained_model(media_selection, args):
    # validity check
    err = 0
    if media_selection == 1 and os.path.splitext(args["input"])[1] not in ['.jpg', '.png', '.gif'] or \
            media_selection == 2 and os.path.splitext(args["input"])[1] not in ['.mp4', '.gif']:
        print("error media chosen, try again")
        err = -1

    if not err:
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label

        # derive the paths to the YOLO weights and model configuration
        weightspath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        configpath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return ln, net, LABELS, err


def load_input_snap(selected_media):
    # load media file
    vs = None
    image = None
    (H, W) = (None, None)

    if selected_media == 1:
        # load our input image and grab its spatial dimensions
        image = cv2.imread(args["input"])
        if image and image.any():
            (H, W) = image.shape[:2]
    else:
        # load our input streaming video and grab its spatial dimensions
        vs = cv2.VideoCapture(args["input"])

    return image, vs, (H, W)


def load_input_video(vs):
    # handle frame data
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    return total


def show_media_box_boundary(selected_media, ln, snap):
    # construct a blob from the input image/frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(snap, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    if selected_media == 1:
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    return layer_outputs, start, end


def filter_detected_objects(layer_outputs, LABELS, W, H):
    # filter relevant objects by class, threshold , confidence
    # initialize our lists respectively
    boxes = []  # lists of detected bounding boxes
    confidences = []  # confidences
    class_ids = []  # class IDs

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence of current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability (confidence determined)
            if 'car' == LABELS[class_id]:
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the snap, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    return idxs, boxes, confidences, class_ids


def bound_box_to_json(collect_data, j, x, y, width, high):
    # build chosen objects list's data for json file
    collect_data.append({
        'detected_obj': j,
        'x_cord': x,
        'y_cord': y,
        'x_cord+width': width,
        'y_cord+high': high
    })

    return collect_data


def output_detections_and_confidence(selected_media, current_media_file, idxs, snap, boxes, confidences, class_ids):
    # extract the bounding box coordinates
    data = {}
    data[current_media_file] = []

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        j = 1  # detected car bounding box numerator
        color = (0, 0, 255)  # align the detected bounding boxes in red color

        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            text = "{}{}: {:.4f}".format(LABELS[class_ids[i]], j, confidences[i])

            # draw a bounding box rectangle and label on the snap - image or frame
            cv2.rectangle(snap, (x, y), (x + w, y + h), color, 2)
            cv2.putText(snap, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            if selected_media == 1:
                # collect item's bound box to save in a json file
                data[current_media_file] = bound_box_to_json(data[current_media_file], j, x, y, x + w, y + h)
            j += 1

        return data[current_media_file]


def handle_isolated_frames(vs, ln, total, current_media_file):
    # loop over frames from the video file stream
    (H, W) = (None, None)
    # initialize the video stream, pointer to output video file
    writer = None

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

            layer_outputs, start, end = show_media_box_boundary(2, ln, frame)

            idxs, boxes, confidences, class_ids = filter_detected_objects(layer_outputs, LABELS, W, H)

            output_detections_and_confidence(2, current_media_file, idxs, snap, boxes, confidences, class_ids)

            print(writer)

            # check if the video writer is None
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
                # some information on processing single frame
                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * total))
            # write the output frame to disk
            writer.write(frame)

    return writer


def handle_video_frames(vs, ln, total, current_media_file):
    color = (0, 0, 255)  # align the detected bounding boxes in red color
    (W, H) = (None, None)
    writer = None
    data = {}
    data[current_media_file] = []

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

            # construct a blob from the input frame add our bounding boxes
            # and associated probabilities

            layer_outputs, start, end = show_media_box_boundary(2, ln, frame)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            idxs, boxes, confidences, class_ids = filter_detected_objects(layer_outputs, LABELS, W, H)

            data[current_media_file] = output_detections_and_confidence(2, current_media_file, idxs, frame, boxes,
                                                                        confidences,
                                                                        class_ids)

            # check if the video writer is None
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
                # some information on processing single frame
                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * total))
            # write the output frame to disk
            writer.write(frame)

    return writer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Would you like to detect a car out of image or a video?")
    user_media_selection = int(input(" Choose for image 1 and for video 2. Default choose is image (1)"))

    # default value
    if user_media_selection != 1 or user_media_selection != 2:
        user_media_selection = 1

    # construct the argument parse and parse the arguments
    args = construct_arguments(user_media_selection)

    # load class labels & weights & config our YOLO model was trained on
    ln, net, LABELS, err = load_trained_model(user_media_selection, args)

    if not err:

        # load input image & video and grab its spatial dimensions
        # and output associated probabilities
        # load input image  and grab its spatial dimensions
        snap, vs, (H, W) = load_input_snap(user_media_selection)

        if user_media_selection == 1:
            # image output bounding boxes and associated probabilities
            layerOutputs, start, end = show_media_box_boundary(1, ln, snap)

            # loop over each of detected item
            idxs, boxes, confidences, class_ids = filter_detected_objects(layerOutputs, LABELS, W, H)

            # extract each detected object bounding box coordinates
            data = output_detections_and_confidence(1, os.path.basename(args["input"]), idxs, snap, boxes, confidences,
                                                    class_ids)

            # save item's bounding boxes to json file - per specific image
            name = os.path.splitext(os.path.basename(args["input"]))
            outputPath = os.path.sep.join([args["output"], str(name)])
            suffix = '.txt'
            file_name = name[0] + suffix
            with open(file_name, 'w') as outfile:
                json.dump(data, outfile)

            # show the output image
            cv2.imshow("Image", snap)
            cv2.waitKey(0)
        else:
            # load input video and grab its spatial dimensions
            total = load_input_video(vs)
            writer = handle_video_frames(vs, ln, total, os.path.basename(args["input"]))

            # release the file pointers
            print("[INFO] cleaning up...")
            if writer:
                writer.release()
                vs.release()


