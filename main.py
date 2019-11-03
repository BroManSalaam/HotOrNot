import tensorflow as tf
import numpy as np
import faces
import model
import pandas as pd
import cv2
from tensorflow.python.client import device_lib
import os

def main():
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer("IMG_SIZE", 200,
                                """Size of the image to be used for processing""")

    tf.app.flags.DEFINE_integer("n_epochs", 30,
                               """Number of epochs to train for""")

    tf.app.flags.DEFINE_string("data_dir", "data/",
                               """Directory that will contain data for testing and training""")

    # img_array = cv2.imread(os.path.join("data/images/AF1.jpg"))
    #
    # img_array = faces.crop_face(img_array)
    #
    # cv2.imshow("image", img_array)
    # cv2.waitKey(0)

    # faces.slideshow_pickle()
    images, scores = faces.load(generate_new=False, slideshow=False, demographics=["CF"])
    cn = model.ConvolutionalNetwork()
    cn.train_network(images[0], scores[0], "CF_hot_or_not.model")
    print(cn.get_custom_predictions("CF_hot_or_not.model"))

if __name__ == "__main__":
    main()
