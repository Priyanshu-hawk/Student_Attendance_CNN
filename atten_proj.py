import numpy as np
import face_recognition
import cv2
import datetime
import os
from PIL import Image
import numpy as np
import pickle
import make_encodings

print("[+] Loading Model")

if not os.path.exists('stud.pkl'):
    print("[+] stud.pkl not found, running encoding")
    make_encodings.startEncoding()

with open('stud.pkl', 'rb') as f:
    data_record = pickle.load(f)

imgs = data_record['imgs']
cls_name = data_record['cls_name']
face_encodings = data_record['face_encodings']
print("[+] Model intialize")

def check_new_img(cls_name):
    base_dir = 'img'
    img_list = os.listdir(base_dir)
    cls_name_now = []
    for idx,cls in enumerate(img_list):
        cls_name_now.append(cls.split('.')[0])
    if len(cls_name)!=len(cls_name_now):
        print("[+] Found Changes in img folder, running encoding")
        make_encodings.startEncoding()

check_new_img(cls_name)

def make_attendence_csv():
    now = datetime.datetime.now().strftime("%d_%m_%y")
    file_name = 'attendence'+'_'+now+'.csv'
    if not os.path.exists(file_name):
        print("[+] Makeing",file_name)
        with open(file_name,'w') as f:
            f.writelines('name,time')
    
    return(file_name)
    
attendence_file = make_attendence_csv()

def mark_attend(name):
    with open(attendence_file, 'r+') as f:
        my_data = f.readlines()
        exe_name = []
        for line in my_data:
            exe_name.append(line.split(',')[0])
        
        if name not in exe_name:
            now = datetime.datetime.now().strftime('%H:%M:%S')
            f.writelines(f'\n{name},{now}')


cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    img = cv2.resize(img,(0,0), None, 0.50, 0.50)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_current = face_recognition.face_locations(img, model='cnn') # for gpu based machines
    # face_current = face_recognition.face_locations(img, model='hog') # for cpu based machines
    encd_current = face_recognition.face_encodings(img, face_current)


    for encd_face, face_loc in zip(encd_current, face_current):
        matches = face_recognition.compare_faces(face_encodings, encd_face, tolerance=0.5)
        faceDist = face_recognition.face_distance(face_encodings, encd_face)
        
        fd = np.argmin(faceDist)
        try:
            mch = matches.index(True)
        except:
            mch = -1
        if fd==mch:
            curr_person = cls_name[fd]
        
            y1,x2,y2,x1 = face_loc
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            
            h,w = cv2.getTextSize(f'{curr_person}', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
            cv2.putText(img, f'{curr_person}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

            mark_attend(curr_person)

    cv2.imshow('Student', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break