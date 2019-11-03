import pandas as pd
import pickle
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = "data/"
pickle_dir = "data/pickles"
X_name = "X.pickle"
y_name = "y_classification.pickle"
img_dir = "img/"
IMG_SIZE = 200

def save_pickle_data(X, y, demographics=["CM", "CF", "AM", "AF"]):

    index = 0

    for demographic in demographics:
        with open(os.path.join(pickle_dir, demographic, X_name), "wb") as pickle_out:
            pickle.dump(X[index], pickle_out)

        with open(os.path.join(pickle_dir, demographic, y_name), "wb") as pickle_out:
            pickle.dump(y[index], pickle_out)

        index += 1

def load_pickle_data(demographics=["CM", "CF", "AM", "AF"]):

    images = []
    scores = []

    for demographic in demographics:
        print("\rLoading pickled data...", demographic, flush=True, end='')

        with open(os.path.join(pickle_dir, demographic, X_name), "rb") as pickle_in:
            images.append(pickle.load(pickle_in))

        with open(os.path.join(pickle_dir, demographic, y_name), "rb") as pickle_in:
            scores.append(pickle.load(pickle_in))

    return images, scores


def crop_face(img):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    if len(faces) == 0:
        return img

    for (x, y, w, h) in faces:
        return img[y:y + h, x:x + w]


# def crop_face(img):
#     face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 1.25, 6)
#
#     if len(faces) is not 1:
#         return img[0:IMG_SIZE, 0:IMG_SIZE]
#     else:
#         for (x, y, w, h) in faces:
#             return faces[y:y + h, x:x + w]
#
#     facemark = cv2.face.createFacemarkLBF()
#     facemark.loadModel("cascades/lbfmodel.yaml")
#
#     for (x, y, w, h) in faces:
#         is_success, landmarks = facemark.fit(img, faces)
#
#         for point in landmarks[0][0]:
#             point = (int(point[0]), int(point[1]))
#             cv2.rectangle(img, (point[0], point[1]), (point[0]+1, point[1]+1), (255, 0, 0), 2)
#
#         return img[y:y + h, x:x + w]

def load(demographics=["CM", "CF", "AM", "AF"], generate_new=False, slideshow=False):

    if generate_new == False:
        return load_pickle_data(demographics)


    data = pd.read_csv(data_dir + "All_Ratings.csv")

    ratings = []

    for demographic in demographics:
        ratings.append(data[data["Filename"].str.startswith(demographic)].groupby("Filename")["Rating"])

    images = []
    scores = []
    index = 0


    print("loading images...")

    for rating_set in ratings:

        demographic_images = []
        demogrpahic_scores = []
        n_images = len(rating_set)

        for filename, score in rating_set:
            try:
                img_array = cv2.imread(os.path.join(data_dir, "images/", filename))
                img_array = crop_face(img_array)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
                img_array /= 255

                if slideshow:
                    print(img_array.shape)
                    show(str(score.mean()), img_array)

                demographic_images.append(img_array)
                demogrpahic_scores.append(score.mean())
            except AttributeError as e:
                print("Attribute error on", filename)
                print(e)
                exit()
            print(f"\rLoading {demographics[ratings.index(rating_set)]} {index / n_images:0.2%}", end='', flush=True)
            index += 1

        images.append(np.array(demographic_images, dtype=np.float32))
        scores.append(demogrpahic_scores)
        index = 0

    print("\nComplete!")

    print("Pickling...")
    save_pickle_data(images, scores, demographics)
    print("Finished")

    return images, scores



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Visualize faces and statistics
#......................................................................................

def show(name, img):
    if type(img) == str:
        img = cv2.imread(str(img))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_average_image(self, images):
    show("Average Image", np.mean(images, axis=0).astype(np.uint16, copy=False))


def slideshow_pickle():
    X, y = load_pickle_data()

    for image, score in zip(X, y):
        show(str(score), image)


def slideshow(race, lower_avg=0, upper_avg=5):
    ratings = pd.read_csv("data/All_Ratings.csv")

    averages = ratings[ratings["Filename"].str.startswith(race)].groupby("Filename")["Rating"].mean()
    target_faces = averages[averages > lower_avg]
    target_faces = target_faces[averages < upper_avg]

    for filename, score in target_faces.iteritems():
        img = cv2.imread("data/images/" + filename)
        cv2.imshow(str(filename) + " " + str(score), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_attractiveness_by_demographic():
    ratings = pd.read_csv("data/All_Ratings.csv")

    demographics = list(ratings["Filename"].str[:2].unique())

    colors = ["pink", "blue", "red", "black"]

    info = pd.DataFrame(index=demographics, columns=["mean", "stddev"])

    for index, demographic in enumerate(demographics):
        group = ratings[ratings["Filename"].str.startswith(demographic)]
        averages = group.groupby("Filename")["Rating"].mean().round().tolist()

        info.loc[demographic]["mean"] = sum(averages) / len(averages)
        info.loc[demographic]["stddev"] = np.std(averages)

        y = [averages.count(i) / len(averages) for i in range(1, 6)]
        plt.plot(range(1, 6), y, color=colors[index], label=demographic)

    print(info)
    plt.legend()
    plt.show()




