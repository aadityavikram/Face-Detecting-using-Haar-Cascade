import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

people = ["", "Emilia Clarke", "Aaditya Vikram", "Kit Harrington", "Nikolaj Coster Waldau", "Peter Dinklage"]

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (550, 550))
    # cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def data(path):
    dir = os.listdir(path)
    faces = []
    labels = []
    for name in dir:
        if not name.startswith("s"):
            continue
        label = int(name.replace("s", ""))
        path_new = path + "/" +name
        people_name = os.listdir(path_new)
        for people in people_name:
            if people.startswith("."):
                continue
            image_path = path_new + "/" + people
            img = cv2.imread(image_path)
            print("processing....")
            # cv2.imshow('image', img)
            cv2.waitKey(100)

            face, rect = detect(img)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels, people_name

faces, labels, name = data("images")

#print(np.array(labels))

print("Data prepared")

print(len(faces), len(labels), name, labels)

face_recog = cv2.face.LBPHFaceRecognizer_create()
face_recog.train(faces, np.array(labels))
# face_recog.save("rec.yml")
face_recog.save("rec1.yml")