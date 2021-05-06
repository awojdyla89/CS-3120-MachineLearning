# -*- coding: utf-8 -*-
"""
@author: Adam

"""

def cls(): return print("\033[2J\033[;H", end='')

cls()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import cv2
import os

def load(imagePath_list, verbose=-1):
    # initialize the list of features and labels
    data = []
    labels = []
    
    # iterate over all the images in the KNN folder
    for (i, imagePath) in enumerate(imagePath_list):
        
        # load the image and extract the class label assuming
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        
        # check to see if our preprocessors are not None
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(label)
        
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(imagePath_list)))
            
    # return a tuple of data and labels        
    return np.array(data), np.array(labels)


# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePath_list = list(paths.list_images("../Week8/KNN/animals"))


# initialize the image preprocessor
# load the dataset from disk
# reshape the data matrix
(data, labels) = load(imagePath_list, verbose=500)
data = data.reshape((data.shape[0], 3072))

# shows information on memory consumption of the images
print("[INFO](extra credit) features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))



# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.30, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=6)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))



















