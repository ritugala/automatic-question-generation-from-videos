#imports

# %matplotlib inline # uncomment line for jupyter notebooks

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
# import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


DATA_INPUT_PATH = '/Users/revathivijay/Desktop/QG_Classifier/Dataset/'


# creating image list and corresponding labels for input
def create_image_and_labels():
    images = []
    images_shape = []
    labels = []
    df = pd.read_csv('input.csv')
    for index, item in df.iterrows():
        filename = f"{item['id']}.jpg"
        image = Image.open(os.path.join(DATA_INPUT_PATH, filename)).convert("L")
        images.append(np.asarray(image))
        labels.append(item['class'])
        images_shape.append(np.array(np.asarray(image).shape))
    images_shape = np.array(images_shape)
    median_image_height = np.int(np.median(images_shape[:,0]))
    median_image_width = np.int(np.median(images_shape[:,1]))


    images = [cv2.resize(img,(64,64)) for img in images]
    return images, labels

def create_model():
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    print('-------------MODEL INFORMATION-----------------')
    print(model.summary())
    return model

def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def train_model(model, X_train, y_train, X_test, y_test, epochs=10):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
    print(f"Trained for {epochs} epochs!")
    return history

def predict(model, list_of_images):
    predictions = model.predict(list_of_images)
    print("Predicted!")
    return predictions

def create_train_test_data(data, labels, test_data=0.2):
    idx = (1-test_data) * len(data)
    X_train, y_train = np.array(data[:idx]), np.array(labels[:idx])
    X_test, y_test = np.array(data[idx:]), np.array(labels[idx:])

    # reshaping images to 64*64*1 for model input
    X_train = X_train.reshape(len(X_train),64,64,1)
    X_test = X_test.reshape(len(X_test),64,64,1)

    print(f'Shape of input image: {X_train_graph[0].shape}')
    return X_train, y_train, X_test, y_test

def shuffle_data(images, labels):
    import sklearn
    array1_shuffled, array2_shuffled = sklearn.utils.shuffle(images, labels)
    return array1_shuffled, array2_shuffled

def prediction_scaling(predictions):
    for i, predicted in enumerate(predictions):
    if predicted[0] > 0.25:
        print "bigger than 0.25"
        #assign i to class 1
    else:
        print "smaller than 0.25"
        #assign i to class 0
