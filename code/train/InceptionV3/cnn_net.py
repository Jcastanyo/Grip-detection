# CODE USED TO TRAIN INCEPTIONV3 MODEL TO DETECT CONTACT IN TACTILE IMAGES

# IMPORT LIBRARIES

import json
import tensorflow as tf 
import numpy 
import h5py
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
import time

# WE USE A TO MEASURE TRAINING TIME
a = 0
# WE USE THIS VARIABLE TO SAVE LOGS
logs_list = []
# WE USE THIS CLASS TO CREATE OUT CUSTOM CALLBACK
class CustomSaver(tf.keras.callbacks.Callback):
    # THIS FUNCION IS EXECUTED AT THE END OF THE EPOCH, WHEN WE WANT TO SAVE MODELS, LOGS, ETC.
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            
            self.model_json = model.to_json()
            
            with open("/commandia/docker/commandia/contact/models/model/model-epoch-{}.json".format(epoch),'w') as json_file:
                json_file.write(self.model_json)
            
            self.model.save_weights("/commandia/docker/commandia/contact/models/weights/weights-epoch-{}.h5".format(epoch))
            print("\nSaved model to disk")

        if epoch % 1 == 0:
            b = time.time()
            c = b-a
            print("\n{} seconds of training".format(c))
            with open("logs.json", "w") as file_pi:
                logs_aux_list = []
                logs_aux_list.append(logs)
                logs_list.append(logs_aux_list)
                json.dump(logs_list, file_pi)


# MAIN FUNCTION
if __name__ == '__main__':
    
    # HERE WE LOAD TRAIN, VALIDATION AND TEST DATASETS.

    f = h5py.File("/commandia/docker/commandia/contact/dataset_sensor50/train_contacto_sensor50.h5", 'r')
    data_train = f['data'][:]
    label_train = f['label'][:]
    f.close()

    f2 = h5py.File("/commandia/docker/commandia/contact/dataset_sensor50/validation_contacto_sensor50.h5", 'r')
    data_validation = f2['data'][:]
    label_validation = f2['label'][:]
    f2.close()
    
    f3 = h5py.File("/commandia/docker/commandia/contact/dataset_sensor50/test_contacto_sensor50.h5", 'r')
    data_test = f3['data'][:]
    label_test = f3['label'][:]
    f3.close()

    
    # WE USE IMAGEDATAGENERATOR FROM KERAS TO APPLY DATA AUGMENTATION
    

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        zoom_range=0.2,
                                        rotation_range=5,
                                        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        zoom_range=0.2,
                                        rotation_range=5,
                                        horizontal_flip=True)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


    train_generator = train_datagen.flow(data_train, label_train,
                                            batch_size=128, shuffle=True)

    validation_generator = validation_datagen.flow(data_validation, label_validation,
                                            batch_size=128, shuffle=True)

    test_generator = test_datagen.flow(data_test, label_test,
                                            batch_size=128, shuffle=True)                                            

    # LOAD INCEPTIONV3 MODEL WITHOUT FINAL CLASSIFIER, WE WANT TO TRAIN USING TRANSFER LEARNING
    
    iv3_model = InceptionV3(include_top=False, weights='imagenet',
                                input_shape=(320, 240, 3))

    # WE APPLY TRANSFER LEARNING, WE FREEZE SOME LAYERS AND TRAIN OTHERS
    
    for layer in iv3_model.layers[:249]:
        layer.trainable = False

    for layer in iv3_model.layers[249:]:
        layer.trainable = True

    # INITIAL MODEL SUMMARY
        
    iv3_model.summary()
    
    # CREATE NEW MODEL AND ADD SOME FINAL LAYERS TO ADAPT THE MODEL TO OUR TASK
    
    model = tf.keras.models.Sequential()
    model.add(iv3_model)

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.6))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.6))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # HYPER-PARAMS
    
    model.compile(loss='binary_crossentropy', 
                    optimizer=tf.keras.optimizers.RMSprop(lr=0.00001),
                    metrics=['accuracy'])

    # FINAL SUMMARY
    model.summary()

    # CREATE AN OBJECT OF OUR CUSTOM CALLBACK
    saver = CustomSaver()

    # START COUNTING TIME
    a = time.time()

    # START TRAINING
    history = model.fit(train_generator, validation_data=validation_generator,
                                    epochs = 100, callbacks=[saver])

    
