import face_recognition
import cv2
import os
from PIL import Image
import numpy as np
import pickle

def startEncoding():
    base_dir = 'img'
    imgs = []
    cls_name = []
    img_list = os.listdir(base_dir)
    for idx,cls in enumerate(img_list):
        imgs.append(cv2.imread(base_dir+'/'+cls))
        cls_name.append(cls.split('.')[0])

    print(cls_name)

    def find_encoding(imgs):
        encds = []
        for i in imgs:
            ig = Image.fromarray(i)
            print(ig.mode)
            encd = face_recognition.face_encodings(i, num_jitters=100, model='large')[0]
            encds.append(encd)
        return encds

    face_encodings = find_encoding(imgs)

    data_record = {'imgs':imgs, 'cls_name':cls_name, 'face_encodings': face_encodings}

    with open('stud.pkl', 'wb') as f:
        pickle.dump(data_record, f)