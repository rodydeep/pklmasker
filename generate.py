import streamlit as st
import cv2
import numpy as np
import requests
from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
from mmdet.utils.contextmanagers import concurrent
from pprint import pprint
from PIL import Image
import datetime



# Specify the path to model config and checkpoint file
config_file = 'configs/fasterrcnn.py'
checkpoint_file = 'models/fasterrcnn.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo2.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

list_objects = []

for i in result[1]:
    temp = i
    temp = np.append(temp, 1)
    list_objects.append(temp)

for i in result[2]:
    temp = i 
    temp = np.append(temp, 2)
    list_objects.append(temp)

for i in result[3]:
    temp = i
    temp = np.append(temp, 3)
    list_objects.append(temp)

img = cv2.imread(img)
for i in list_objects:
    if i[5] == 1:
        color = (255, 0, 0)
        text  = "Mask weared incorrect"
    elif i[5] == 2:
        color = (0, 255, 0)
        text  = "With mask"
    elif i[5] == 3:
        color = (0, 0, 255)
        text = "Without mask"
    text += ": " + str(round(i[4], 2))
    x1 = i[0]
    y1 = i[1]
    x2 = i[2] - 1
    y2 = i[3] - 1

    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    img = cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite('Original_result.jpg', img)

def IoU(bbox1, bbox2):

    x1_left = bbox1[0]
    y1_top = bbox1[1]
    x1_right = bbox1[2]
    y1_bot = bbox1[3]

    x2_left = bbox2[0]
    y2_top = bbox2[1]
    x2_right = bbox2[2]
    y2_bot = bbox2[3]

    x_left = max(x1_left, x2_left)
    x_right = min(x1_right, x2_right)
    y_top = max(y1_top, y2_top)
    y_bot = min(y1_bot, y2_bot)

    inter = (x_right - x_left) * (y_bot - y_top)
    if x_right < x_left or y_bot < y_top:
        return 0.0
    area1 = (x1_right - x1_left) * (y1_bot - y1_top)
    area2 = (x2_right - x2_left) * (y2_bot - y2_top)
    union = area1 + area2 - inter

    IoU = inter / union
    return IoU

total_people = 0
incorrect = 0
withmask = 0
withoutmask = 0
list_objects = []
isRemove = []
for i in result[1]:
    temp = i
    temp = np.append(temp, 1)
    list_objects.append(temp)
    isRemove.append(0)

for i in result[2]:
    temp = i 
    temp = np.append(temp, 2)
    list_objects.append(temp)
    isRemove.append(0)

for i in result[3]:
    temp = i
    temp = np.append(temp, 3)
    list_objects.append(temp)
    isRemove.append(0)

for i in range(len(list_objects) - 1):
    for j in range(i + 1, len(list_objects)):
        bbox1 = [list_objects[i][0], list_objects[i][1], list_objects[i][2], list_objects[i][3]]
        bbox2 = [list_objects[j][0], list_objects[j][1], list_objects[j][2], list_objects[j][3]]
        if abs(IoU(bbox1, bbox2)) > 0.7:
            if list_objects[i][4] > list_objects[j][4]:
                isRemove[j] = 1
            else:
                isRemove[i] = 1
            # print("IoU", abs(IoU(bbox1, bbox2)))
        

        if list_objects[i][4] < 0.4:
            isRemove[i] = 1
        if list_objects[j][4] < 0.4:
            isRemove[j] = 1

selected_list = []
for i in range(len(list_objects)):
    if isRemove[i] == 0:
        selected_list.append(list_objects[i])

for i in selected_list:
    if i[5] == 1:
        incorrect += 1
    elif i[5] == 2:
        withmask += 1
    elif i[5] ==3:
        withoutmask += 1
    
total_people += incorrect + withmask + withoutmask

img = 'demo2.png'  # or img = mmcv.imread(img), which will only load it once

img = cv2.imread(img)
for i in selected_list:
    if i[5] == 1:
        color = (255, 0, 0)
        text  = "Mask weared incorrect"
    elif i[5] == 2:
        color = (0, 255, 0)
        text  = "With mask"
    elif i[5] == 3:
        color = (0, 0, 255)
        text = "Without mask"
    text += ": " + str(round(i[4], 2))
    x1 = i[0]
    y1 = i[1]
    x2 = i[2] - 1
    y2 = i[3] - 1

    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    img = cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite('New_result.jpg', img)