import cv2
import face_recognition
import os
import numpy as np
import datetime

path = 'Images'
images = []
names = []
cam = cv2.VideoCapture(0)

images_name = os.listdir(path)
print(images_name)

for i in images_name:
    cur_image = cv2.imread(f'{path}/{i}')
    images.append(cur_image)
    names.append(os.path.splitext(i)[0])
print(names)


def encoding(images):
    encode_list =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def attendance(name):
    with open('attendance.csv','r+') as f:
        attendance_list = f.readlines()
        nameList = []
        for line in attendance_list:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time = datetime.datetime.now()
            f.writelines(f'\n{name},{time}')


known_faces = encoding(images)
print('Encoding Complete')


while True:
    ret,img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces_from_cam = face_recognition.face_locations(img)

    encoding_from_cam = face_recognition.face_encodings(img,faces_from_cam)

    for en,faloc in zip(encoding_from_cam,faces_from_cam):
        validate = face_recognition.compare_faces(known_faces,en)
        faceDis = face_recognition.face_distance(known_faces,en)
        print(faceDis)
        match = np.argmin(faceDis)

        if validate[match]:
            person_name = names[match]
            print(person_name)
            y1,x2,y2,x1 = faloc
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img,person_name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendance(person_name)

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Camera', img)

cam.release()
cv2.destroyAllWindows()

