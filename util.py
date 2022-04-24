
import numpy as np
import json
import base64
import cv2
import dlib
import pickle
from tensorflow import keras
from feature_extraction import extractor
from collections import Counter

__class_name_to_number = {}
__class_number_to_name = {}

__SVM, __LOG, __RF = None, None, None

def classify_image(image_base64_data):
    size =  60
    imgs = get_cropped_image_if_2_eyes(image_base64_data)
    result = []
    who = None
    if imgs:
        for img in imgs:
            resized_img = np.array(cv2.resize(img, (size, size))/255.0)
            extracted_features = extractor.predict(resized_img.reshape(-1, 60, 60, 3)).reshape(1, -1)
            svm_pred = __SVM.predict(extracted_features)[0]
            svm_prob = [np.around(prob*100) for prob in __SVM.predict_proba(extracted_features)[0]]
            log_pred = __LOG.predict(extracted_features)[0]
            log_prob = [np.around(prob*100) for prob in __LOG.predict_proba(extracted_features)[0]]
            rf_pred = __RF.predict(extracted_features)[0]
            rf_prob = [np.around(prob*100) for prob in __RF.predict_proba(extracted_features)[0]]
            # print(log_prob)
            # print(svm_prob)
            # print(rf_prob)
            # print(log_pred)
            # print(svm_pred)
            # print(rf_pred)
            counts = Counter([svm_pred, log_pred, rf_pred])
            svm_max = max(svm_prob)
            log_max = max(log_prob)
            rf_max = max(rf_prob)
            svm_argmax = np.argmax(svm_prob)
            log_argmax = np.argmax(log_prob)
            rf_argmax = np.argmax(rf_prob)
            for label, vote in counts.items():
                if vote >= 2:
                    result.append({'idol': class_number_to_name(label),
                            'class': int(label)})
                    break

            if result == []:
                labels_probs = [(svm_pred, svm_max), (log_pred, log_max), (rf_pred, rf_max)]
                who = max([svm_max, log_max, rf_max])
                for tup in labels_probs:
                    if tup[1] == who:
                        result.append({'idol': class_number_to_name(tup[0]),
                                'class': int(tup[0])})

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __SVM, __LOG, __RF
    if __SVM is None and __LOG is None and __RF is None:
        __SVM = pickle.load(open('./artifacts/CNN_SVM', 'rb'))
        __LOG = pickle.load(open('./artifacts/CNN_LOG', 'rb'))
        __RF = pickle.load(open('./artifacts/RF_model', 'rb'))


    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_base64_data):
    face_cascade = cv2.CascadeClassifier('D:/blackpink/Server/haarcascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:/blackpink/Server/haarcascade/haarcascade_eye.xml')

    eyes_li = []
    roi_color_li = []
    try:
        img = get_cv2_image_from_base64_string(image_base64_data)
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >=2:
                eyes_li.append(eyes)
                roi_color_li.append(roi_color)
        return roi_color_li
    except Exception as e:
        print(e)
        return []

if __name__ == '__main__':
    load_saved_artifacts()
