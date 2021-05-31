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



def setup_digit():

    logging.basicConfig(level=logging.DEBUG)

    # Print a list of connected DIGIT's
    digits = DigitHandler.list_digits()
    print("Connected DIGIT's to Host:")
    pprint.pprint(digits)

    # Connect to a Digit device with serial number with friendly name
    digit = Digit("D00050", "Left Gripper")
    digit_cap = DigitHandler.find_digit("D00050")
    digit.connect()

    # Print device info
    print(digit.info())

    # Change LED illumination intensity
    digit.set_intensity(0)
    time.sleep(1)
    digit.set_intensity(255)

    # Change DIGIT resolution to QVGA
    qvga_res = DigitHandler.STREAMS["QVGA"]
    digit.set_resolution(qvga_res)

    # Change DIGIT FPS to 15fps
    fps_30 = DigitHandler.STREAMS["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps_30)
    
    # Find a Digit by serial number and connect manually
    #digit = DigitHandler.find_digit("D00045")
    #pprint.pprint(digit)

    return digit



if __name__ == '__main__':
    digit = setup_digit()

    json_file = open("model-epoch-10.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights-epoch-10.h5")

    model.summary()
    cont_start = 0
    while(True):
        
        # Grab single frame from DIGIT
        frame = digit.get_frame()
        cv2.imshow("ventana", frame)
        cv2.waitKey(1)
        '''
        if cont_start < 40:
            cont_start += 1
            continue
        '''
        cv2.imshow("ventana", frame)
        cv2.waitKey(20)
        frame = np.reshape(frame, (1, 320, 240, 3))
        frame = preprocess_input(frame)
        pred = model.predict(frame)
        
        if pred > 0.5:
            print("no contacto: {}".format(pred))
        else: 
            print("contacto: {}".format(pred))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
