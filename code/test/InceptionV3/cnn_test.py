# THIS CODE IS USED TO DETECT CONTACT IN TACTILE VISUAL IMAGES FROM DIGIT SENSORS

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
from tensorflow.keras.applications.inception_v3 import preprocess_input


# THIS FUNCTION GETS IMAGES FROM DIGIT SENSOR.
def setup_digit():

    logging.basicConfig(level=logging.DEBUG)

    # PRINT A LIST OF CONNECTED DIGIT SENSORS
    digits = DigitHandler.list_digits()
    print("Connected DIGIT's to Host:")
    pprint.pprint(digits)

    # CONNECT TO A DIGIT DEVICE WITH A FRIENDLY SERIAL NAME
    digit = Digit("D00050", "Left Gripper")
    digit_cap = DigitHandler.find_digit("D00050")
    digit.connect()

    # PRINT DEVIDE INFO
    print(digit.info())

    # CHANGE LED ILLUMATION
    digit.set_intensity(0)
    time.sleep(1)
    digit.set_intensity(255)

    # CHANGE DIGIT RESOLUTION TO QVGA
    qvga_res = DigitHandler.STREAMS["QVGA"]
    digit.set_resolution(qvga_res)

    # SET DIGIT FPS
    fps_30 = DigitHandler.STREAMS["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps_30)
    
    return digit



if __name__ == '__main__':
    # CONFIG DIGIT SENSOR
    digit = setup_digit()

    # LOAD ARCHITECTURE AND WEIGHTS OF THE TRAINED MODEL
    json_file = open("model-epoch-10.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights-epoch-10.h5")

    # MODEL SUMMARY
    model.summary()
    
    #INFINITE LOOP
    while(True):
        
        # GRAB SINGLE FRAME FROM DIGIT SENSOR
        frame = digit.get_frame()
        
        # SHOW THE IMAGE
        cv2.imshow("window", frame)
        cv2.waitKey(1)
      
        # IMAGE PREPROCESSING TO FEED THE CNN
        frame = np.reshape(frame, (1, 320, 240, 3))
        frame = preprocess_input(frame)
        # PREDICTING
        pred = model.predict(frame)
        
        # APPLY THRESHOLD TO GET THE FINAL CLASSIFICATION
        if pred > 0.5:
            print("No contact: {}".format(pred))
        else: 
            print("Contact: {}".format(pred))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
