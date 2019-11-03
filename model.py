import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
import cv2
import os
import faces
import matplotlib.pyplot as plt
import numpy as np
import datetime

model_dir = "models/"


class ConvolutionalNetwork(object):

    def __init__(self):
        self.hot_threshold = 3
        self.FLAGS = tf.app.flags.FLAGS

    def train_network(self, X, y, file_name):
        y = np.array(y)

        # for image, score in zip(X, y):
            # faces.show(str(score), image)

        model = Sequential()

        model.add(tf.keras.layers.Conv2D(128, (32, 32), kernel_initializer='he_normal', input_shape=X.shape[1:]))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))

        model.add(tf.keras.layers.Conv2D(128, (24, 24), kernel_initializer='he_normal'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(256, (10, 10), kernel_initializer='he_normal'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(256, activation=tf.nn.relu))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.4))
        model.add(Dense(1))

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min",
                                                          verbose=1, patience=2, restore_best_weights=True)

        optimizer = tf.keras.optimizers.Adam(lr=0.001)

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error'])

        history = model.fit(X, y, batch_size=16, epochs=self.FLAGS.n_epochs,
                  validation_split=0.2, callbacks=[early_stopping])

        print("saving to", os.path.join(model_dir, file_name), "...")
        model.save(os.path.join(model_dir, file_name))

        print("finished")

        # plt.plot(range(len(history.history["val_loss"])), history.history["val_loss"][:self.FLAGS.n_epochs], color="red")
        # plt.xlabel("epochs")
        # plt.ylabel("val_loss")
        #
        # plt.show()

        return model

    def is_hot(self, prediction):
        return False if prediction <= self.hot_threshold else True


    def prep_img(self, path):
        img_array = cv2.imread(path)
        img_array = faces.crop_face(img_array)
        img_array = cv2.resize(img_array, (self.FLAGS.IMG_SIZE, self.FLAGS.IMG_SIZE)).astype(np.float32)
        img_array /= 255
        return img_array.reshape(-1, self.FLAGS.IMG_SIZE, self.FLAGS.IMG_SIZE, 3)

    def get_custom_predictions(self, demographics):
        test_lst = os.listdir("test/custom/CF/")

        print("running predictions on", len(test_lst), "images for ", demographics)

        for demographic in demographics:
            file_names = test_lst = os.listdir(os.path.join("test/custom/", demographic))

            predictions = dict()
            try:
                model = tf.keras.models.load_model(os.path.join(model_dir, (demographic + ".model")))
            except FileNotFoundError:
                print(os.path.join(model_dir, (demographic + ".model")), "not found")
    
            for file in test_lst:
                try:
                    img = self.prep_img(os.path.join("test/custom/CF/", file))
                    predictions[file] = model.predict(img)
                    cv2.imshow(str(predictions[file]), img[0])
                    cv2.waitKey(0)
                except Exception as e:
                    print(e)
                    continue

                cv2.destroyAllWindows()

        return predictions




    # def get_custom_predictions(self, file_name):
    #     test_lst = os.listdir("test/custom/CF/")
    #
    #     print("running predictions on", len(test_lst), "images")
    #
    #     predictions = dict()
    #     model = tf.keras.models.load_model(os.path.join(model_dir, file_name))
    #
    #     for file in test_lst:
    #         try:
    #             img = self.prep_img(os.path.join("test/custom/CF/", file))
    #             predictions[file] = model.predict(img)
    #             cv2.imshow(str(predictions[file]), img[0])
    #             cv2.waitKey(0)
    #         except Exception as e:
    #             print(e)
    #             continue
    #
    #         cv2.destroyAllWindows()
    #
    #     return predictions


