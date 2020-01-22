# [ Load Package ]
# ----------------------
import os
import argparse
import joblib

import keras
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger,  TerminateOnNaN
from sklearn.model_selection import train_test_split

from metrics import *
from scheduler import *
import tensorflow.compat.v1 as tf
from cv2 import cv2
from keras.utils import np_utils, to_categorical
import myDataGen as dg

import archs
# ----------------------

# [ Target Data_China Face_Noeun ]
# ----------------------
TRAIN_DIR = str(os.getcwd())
info = pd.read_csv('face_embedding_nn_exp1_train.csv')

data = []
labels = []

for i in range(len(info)-37000):
    img = cv2.imread(info.iloc[i]['file_name'])
    label = (info.iloc[i]['label'])-1
    data.append(img)
    labels.append(label)
    print(i)

# 픽셀 [0, 1]로 만들기 위함
data = np.array(data, dtype="float") / 255.0
labels = np_utils.to_categorical(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)
trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)


# DataGenerator
aug = dg.DataGenerator(data, labels)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

print('done')
# ----------------------


# [ GPU Setting ]
# ----------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto() # tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sessopm = tf.Session(config=config)
# ----------------------

# vgg8_arcface
arch_names = archs.__dict__.keys()
arch_names = archs.__dict__.keys()

# Options
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg8',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: vgg8)')
    parser.add_argument('--num-features', default=3, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    # ! Adam method : Adagrad + RMSPro Cost Func
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # add model name to args
    args.name = 'mnist_%s_%dd' %(args.arch, args.num_features)

    os.makedirs('models/%s' %args.name, exist_ok=True)

    # print options
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name) # save model as pickled binary file

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    # [ Target Data Generating ]
    # ---------------------- MNIST
    (X, y), (X_test, y_test) = mnist.load_data()

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

    y = keras.utils.to_categorical(y, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    # ----------------------

    # [ Optimizer Setting ]
    # ----------------------
    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)
    # ----------------------

    # [ Model Setting ]
    # ----------------------
    model = archs.__dict__[args.arch](args)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary() # check

    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))
    # ----------------------

    # [ Train ]
    # ----------------------
    if 'face' in args.arch:
        model.fit_generator(aug.__data_geeration([trainX, trainY], trainY),
                            steps_per_epoch=len(trainX) / 32,
                            epochs=args.epochs,
                            validation_data=([testX, testY], trainY))
        '''''''''
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
                    
        model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
        
        model.fit(X, y, validation_data=(X_test, y_test),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
        '''''''''
    else:
        model.fit_generator(aug, steps_per_epoch=len(trainX) / 32,
                            epochs=args.epochs,
                            validation_data=([testX, testY], testY))

        model.fit_generator(trainX, trainY,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            verbose=1)
    # ----------------------

    # [ Model Evaluate ]
    # ----------------------
    model.load_weights(os.path.join('models/%s/model.hdf5' %args.name))
    if 'face' in args.arch:
        score = model.evaluate([X_test, y_test], y_test, verbose=1)
    else:
        score = model.evaluate(X_test, y_test, verbose=1)
    # ----------------------

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

if __name__ == '__main__':
    main()
