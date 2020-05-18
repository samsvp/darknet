#!/usr/bin/env python

from __future__ import print_function, division

import os

# change location to scripts. Remove when ros parameters are implemented
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import rospy
from robosub_msgs.msg import Detection, DetectionArray
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from ctypes import *
import math
import random
# import os
import time
import darknet


_netMain = None
_metaMain = None
_darknet_image = None
_resized_frame = None

def convert_back(x, y, w, h):
    '''
    Converts the parameters of the boxes to the opencv format
    '''
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cv_draw_boxes(detections, img):
    '''
    Draws the detected boxes
    '''
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convert_back(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def load_network():
    '''
    Initializes the darknet
    '''
    # use ros param for it
    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./yolov3-tiny.weights"
    metaPath = "./cfg/coco.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

    metaMain = darknet.load_meta(metaPath.encode("ascii"))

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    return netMain, metaMain, darknet_image


def resize_frame(cap):
    '''
    Resizes the images to darknet's cfg size 
    '''
    ret, frame_read = cap.read()
    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(_netMain),
                                darknet.network_height(_netMain)),
                                interpolation=cv2.INTER_LINEAR)

    return frame_resized


def detect():
    '''
    Returns the detections made by darknet and a ros detection message type
    '''

    darknet.copy_image_from_bytes(_darknet_image, _resized_frame.tobytes())

    detections = darknet.detect_image(_netMain, _metaMain, _darknet_image, thresh=0.25) 

    detection_array = DetectionArray() 

    for d in detections:
        box = Detection()

        box.label = d[0].decode()

        box.x = d[2][0] / darknet.network_width(_netMain) 
        box.y = d[2][1] / darknet.network_height(_netMain) 
        box.w = d[2][2] / darknet.network_width(_netMain)
        box.h = d[2][3] / darknet.network_height(_netMain)

        detection_array.boxes.append(box)
        
    return detections, detection_array
    

def detections_img(cap, detections):
    '''
    Draws the boxes on the camera image  
    '''
    image = cv_draw_boxes(detections, _resized_frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def YOLO():
    '''
    Main YOLO loop 
    '''

    global _netMain, _metaMain, _darknet_image, _resized_frame

    _netMain, _metaMain, _darknet_image = load_network()

    detections_pub = rospy.Publisher('darknet', DetectionArray, queue_size=10)
    image_pub = rospy.Publisher("darknet_image", Image, queue_size=10)

    rospy.init_node('DarknetNode', anonymous=True)

    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)
    
    bridge = CvBridge()

    print("Starting the YOLO loop...")

    while not rospy.is_shutdown():
        prev_time = time.time()

        _resized_frame = resize_frame(cap)

        detections, detection_array = detect()
        detections_pub.publish(detection_array)

        image = detections_img(cap, detections)

        image_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

        print(1/(time.time()-prev_time))
    
    cap.release()
    

if __name__ == "__main__":
    YOLO()
