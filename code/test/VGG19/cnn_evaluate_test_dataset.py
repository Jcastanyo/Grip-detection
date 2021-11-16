# WE USE THIS CODE TO EVALUATE OUR MODEL AND SEE TEST METRICS

# IMPORT LIBRARIES

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

    # LOAD ARCHITECTURE AND WEIGHTS OF THE TRAINED MODEL
    json_file = open("model-epoch-95.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights-epoch-95.h5")

    # MODEL SUMMARY
    model.summary()
    
    # LOAD TEST SET FROM DIRECTORY
    filename = "/home/commandia2/Desktop/COMMANDIA/codigos_pruebas/cnn_contacto/"

    f = h5py.File(filename + "test_contacto_sensores_3recuperan.h5", 'r')
    data_test = f['data'][:]
    label_test = f['label'][:]
    f.close()

    # HERE WE SAVE PREDICTIONS AND TIME VALUES IN LISTS
    predictions = []
    times = []
    
    # PREDICTION AND TIME FOR EACH IMAGE
    for cont, img in enumerate(data_test):
        # START TIME
        a = time.time()
        # IMAGE PREPROCESSING TO FEED THE CNN
        frame = np.reshape(img, (1, 320, 240, 3))
        frame = preprocess_input(frame)
        # PREDICTING
        pred = model.predict(frame)

        # APPLY THRESHOLD TO GET THE FINAL CLASSIFICATION
        if pred > 0.5:
            pred_class = 1
        else: 
            pred_class = 0

        # FINAL TIME
        b = time.time()
        
        # CALCULATE INTERVAL AND SAVE
        times.append((b-a))
        
        # SAVE PREDICTION
        predictions.append(pred_class)

    # PREDICTION IN NP ARRAY TYPE
    predictions_np = np.asarray(predictions)

    # CALCULATE METRICS
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

    # CALCULATE MEAN INFERENCE TIME
    
    aux = 0
    for i in times:
        aux += i
    print("Tiempo medio de inferencia: ", aux/len(times))
    
