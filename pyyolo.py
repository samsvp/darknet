#!/usr/bin/env python

from __future__ import print_function, division

import cv2
from ctypes import *
import math
import random
import os
import time
import darknet


class YOLO:

    def __init__(self, configPath, weightPath, metaPath, thresh=0.25):    
        self.netMain, self.metaMain, self.darknet_image = \
            self.load_network(configPath, weightPath, metaPath)
        self.thresh = thresh
        self.resized_frame = None


    def convert_back(self, x, y, w, h):
        '''
        Converts the parameters of the boxes to the opencv format
        '''
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def cv_draw_boxes(self, detections, img):
        '''
        Draws the detected boxes
        '''
        
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convert_back(
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


    def load_network(self, configPath, weightPath, metaPath):
        '''
        Initializes the darknet
        '''
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


    def resize_frame(self, frame):
        '''
        Resizes the images to darknet's cfg size 
        '''
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resized_frame = cv2.resize(frame_rgb,
                                    (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                    interpolation=cv2.INTER_LINEAR)

        return self.resized_frame


    def detect(self, frame):
        '''
        Returns the detections made by darknet and a ros detection message type
        '''
        self.resize_frame(frame)

        darknet.copy_image_from_bytes(self.darknet_image, self.resized_frame.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain,
                self.darknet_image, thresh=self.thresh) 
            
        return detections
        

    def detections_img(self, detections):
        '''
        Draws the boxes on the camera image  
        '''
        image = self.cv_draw_boxes(detections, self.resized_frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
