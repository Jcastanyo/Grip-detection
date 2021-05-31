import json
import tensorflow as tf 
import numpy 
import h5py
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import time

a = 0
logs_list = []

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            self.model_json = model.to_json()
            with open("/commandia/docker/commandia/contact/models/model/model-epoch-{}.json".format(epoch),'w') as json_file:
                json_file.write(self.model_json)
            #self.model.save("/commandia/docker/commandia/contact/models/model_{}.hd5".format(epoch))
            self.model.save_weights("/commandia/docker/commandia/contact/models/weights/weights-epoch-{}.h5".format(epoch))
            print("\nSaved model to disk")

        if epoch % 1 == 0:
            b = time.time()
            c = b-a
            print("\n{} segundos de entrenamiento".format(c))
            with open("logs.json", "a") as file_pi:
                logs_aux_list = []
                logs_aux_list.append(logs)
                logs_list.append(logs_aux_list)
                json.dump(logs_list, file_pi)



if __name__ == '__main__':

    #filename = "/home/commandia2/Desktop/COMMANDIA/codigos_pruebas/cnn_contacto/"

    f = h5py.File("/commandia/docker/commandia/contact/train_3recuperan.h5", 'r')
    data_train = f['data'][:]
    label_train = f['label'][:]
    f.close()

    f2 = h5py.File("/commandia/docker/commandia/contact/validation_3recuperan.h5", 'r')
    data_validation = f2['data'][:]
    label_validation = f2['label'][:]
    f2.close()

    f3 = h5py.File("/commandia/docker/commandia/contact/test_3recuperan.h5", 'r')
    data_test = f3['data'][:]
    label_test = f3['label'][:]
    f3.close()

    '''
    label_train = tf.keras.utils.to_categorical(label_train, dtype="uint8")
    label_validation = tf.keras.utils.to_categorical(label_validation, dtype="uint8")
    label_test = tf.keras.utils.to_categorical(label_test, dtype="uint8")
    '''

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
                                            batch_size=32, shuffle=True)

    validation_generator = validation_datagen.flow(data_validation, label_validation,
                                            batch_size=32, shuffle=True)

    test_generator = test_datagen.flow(data_test, label_test,
                                            batch_size=32, shuffle=True)                                            

    '''
    for x,y in train_generator:
        for i in x:
            cv2.imshow("ventana", i)
            cv2.waitKey(500)                                            
    '''

    vgg_model = VGG19(include_top=False, weights='imagenet',
                                input_shape=(320, 240, 3))

    #vgg_model.summary()

    for layer in vgg_model.layers[:7]:
        layer.trainable = False

    '''
    for i, layer in enumerate(resnet_model.layers):
        print(i, layer.name, "-", layer.trainable)
    '''

    model = tf.keras.models.Sequential()
    model.add(vgg_model)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', 
                    optimizer=tf.keras.optimizers.Adam(lr=1e-6),
                    metrics=['accuracy'])

    model.summary()

    saver = CustomSaver()
    a = time.time()
    history = model.fit(train_generator, validation_data=validation_generator,
                                    epochs = 100, callbacks=[saver])

