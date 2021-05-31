import numpy as np
import cv2
import tensorflow as tf 
import logging
import pprint
import time
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import h5py


if __name__ == '__main__':

    json_file = open("model-epoch-95.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights-epoch-95.h5")

    model.summary()
    
    filename = "/home/commandia2/Desktop/COMMANDIA/codigos_pruebas/cnn_contacto/"

    f = h5py.File(filename + "test_contacto_sensores_3recuperan.h5", 'r')
    data_test = f['data'][:]
    label_test = f['label'][:]
    f.close()

    print(data_test.shape[0])

    predictions = []
    times = []
    for cont, img in enumerate(data_test):
        a = time.time()
        frame = np.reshape(img, (1, 320, 240, 3))
        frame = preprocess_input(frame)
        pred = model.predict(frame)

        if pred > 0.5:
            pred_class = 1
        else: 
            pred_class = 0

        b = time.time()
        
        times.append((b-a))
        
        predictions.append(pred_class)

    predictions_np = np.asarray(predictions)

    print(confusion_matrix(label_test, predictions_np))
    tn, fp, fn, tp = confusion_matrix(label_test, predictions_np).ravel()
    print(tn, fp, fn, tp)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(label_test, predictions_np)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(label_test, predictions_np)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(label_test, predictions_np)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(label_test, predictions_np)
    print('F1 score: %f' % f1)

    aux = 0
    for i in times:
        aux += i
    print("Tiempo medio de inferencia: ", aux/len(times))
    