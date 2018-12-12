import cv2
from face import detect, face_recog, people
import matplotlib.pyplot as plt

face_recog = cv2.face.LBPHFaceRecognizer_create()
# face_recog.read("rec.yml")
face_recog.read("rec1.yml")

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

def text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect(img)
    label = face_recog.predict(face)
    label_text = people[label[0]]
    draw_rectangle(img, rect)
    text(img, label_text, rect[0], rect[1] - 5)
    return img

print("Predicting images...")
test_img = cv2.imread('images/4.jpg')
predicted_img1 = predict(test_img)
print("Prediction complete")
cv2.imshow("Prediction", predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
