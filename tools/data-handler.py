#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os
import json
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd

from tiny_facedetect.faceDetect import SSDModel, Predictor

root = r'F:\BaiduNetdiskDownload\Manually_Annotated'
target = r'F:\BaiduNetdiskDownload\Manually_Annotated\tiny_facedetect_filter_annotated_images'
image_dir = os.path.join(root, 'Manually_Annotated_Images')
tarin_file = os.path.join(root, 'training.csv')
vaild_file = os.path.join(root, "validation.csv")

emotion_labels=['Neutral','Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# affect_vaild = [d for i, d in pd.read_csv(vaild_file).iterrows()]
affect_train = [d for i, d in pd.read_csv(tarin_file).iterrows()]

device = "cuda"
net = SSDModel(size=128, device=device)
net.load("./tiny_facedetect/slim-320.pth")  # network input size default: 320 optional value 128/160/320/480/640/1280'
predictor = Predictor(net, size=128, device=device)



vaild_data = []
for d in tqdm(affect_train):
    if d.expression >= 8 or d.face_width < 0:
        continue
    target_path = os.path.join(target, d.subDirectory_filePath).replace("/", "\\")

    if os.path.exists(target_path):
        continue

    folder_path = os.path.dirname(target_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # print(d.subDirectory_filePath,d.face_x,d.face_y, d.face_width, d.face_height, d.expression)
    # x1, y1, x2, y2 = int(d.face_x),int(d.face_y),int(d.face_x + d.face_width),int(d.face_x + d.face_height)
    frame = cv2.imread(os.path.join(image_dir, d.subDirectory_filePath).replace("/", "\\"))
    boxes, probs = predictor.predict(frame)

    if boxes.size(0) == 1:
        vaild_json = {}
        box = boxes[0, :]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = frame[y1:y2, x1:x2]

        try:
            cv2.imwrite(target_path, crop)
        except:
            continue

        vaild_json['name'] = d.subDirectory_filePath
        vaild_json['bbox'] = [x1, y1, x2, y2]
        vaild_json['expression'] = int(d.expression)
        vaild_json['valence'] = d.valence
        vaild_json['arousal'] = d.arousal
        vaild_data.append(vaild_json)



with open(os.path.join(root, "tiny_facedetect_train_filter.json"), "w") as json_file:
    json.dump(vaild_data, json_file)

