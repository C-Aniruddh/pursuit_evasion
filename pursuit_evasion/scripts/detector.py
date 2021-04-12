import numpy as np
import sys
import imutils
import time
import cv2
import pathlib


class ObjectDetector:
    def __init__():
        CURRENT_DIRECTORY = pathlib.Path().cwd()
        PATH_MOBILENETS_MODEL = CURRENT_DIRECTORY.joinpath('weights').joinpath('MobileNetSSD_deploy.caffemodel')
        PATH_MOBILENET_PROTO = CURRENT_DIRECTORY.joinpath('weights').joinpath('MobileNetSSD_deploy.prototxt.txt')

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.THRESHOLD = 0.2

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(str(PATH_MOBILENET_PROTO.resolve()), str(PATH_MOBILENETS_MODEL.resolve()))

        # Model Loaded
        print("[INFO] Model loaded...")


    def process_single_image(self, frame)
        # frame = imutils.resize(frame, width=800)
            
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        result_frame = frame.copy()

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = net.forward()

        output_dict = {}
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.THRESHOLD:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if self.CLASSES[idx] == "person":
                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(self.CLASSES[idx],
                        confidence * 100)
                    cv2.rectangle(result_frame, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(result_frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                    output_dict['bbox'] = {
                        'startX': startX,
                        'startY': startY,
                        'endX': endX,
                        'endY': endY
                    }
                    output_dict['confidence'] = confidence * 100
                    output_dict['re_frame'] = result_frame

        retval = False
        if len(list(output_dict.keys())) > 0:
            retval = True

        return retval, output_dict