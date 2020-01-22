import os
import numpy as np
from cv2 import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

TRAIN_DIR = str(os.getcwd())+r'/MNIST/trainingSet'
train_folder_list = array(os.listdir(TRAIN_DIR))

train_input = []
train_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출
integer_encoded = label_encoder.fit_transform(train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        print(img_path, ": ", img) # Debugging
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_input.append([np.array(img)])
        train_label.append([np.array(onehot_encoded[index])])

print("[ 1 ]", train_input)
train_input = np.reshape(train_input, (-1, 784))
train_label = np.reshape(train_label, (-1, 4))
print("[ 2 ]", train_input)

train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)

np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)