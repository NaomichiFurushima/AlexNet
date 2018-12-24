import os, sys
import cv2
import random
import numpy as np
import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import ticker

from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, TensorBoard, CSVLogger
from keras.utils.np_utils import to_categorical

np.random.seed(1224)

ROWS = 224
COLS = 224
CHANNELS = 3

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'
CACHE_DIR = '../data/cache/'
TRAIN_CACHE_DIR = CACHE_DIR + 'train/'
TEST_CACHE_DIR = CACHE_DIR + 'test/'

FORCE_CONVERT = False

def read(name, reshape=False):
    ret = cv2.imread(name, cv2.IMREAD_COLOR)
    if reshape == True:
        ret = ret.reshape(ret.shape)
    return ret

def convert(img):
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    conv = img.reshape(img.shape)
    return img, conv

def save(name, img):
    cv2.imwrite(name, img)
    return img

def ls(dirname):
    return [dirname + i for i in os.listdir(dirname)]

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
if not os.path.exists(TRAIN_CACHE_DIR):
    os.mkdir(TRAIN_CACHE_DIR)
if not os.path.exists(TEST_CACHE_DIR):
    os.mkdir(TEST_CACHE_DIR)

sys.stdout.write('loading...')
sys.stdout.flush()

train = []
test = []
train_files = ls(TRAIN_CACHE_DIR)
for i in train_files:
    if 'jpg' in i:
        train.append(read(i, True))

test_files = ls(TEST_CACHE_DIR)
for i in test_files:
    if 'jpg' in i:
        test.append(read(i, True))

print('Done!')

if FORCE_CONVERT or len(train) < 25000:
    sys.stdout.write('Process train data...')
    sys.stdout.flush()
    for i in os.listdir(TRAIN_DIR):
        if 'jpg' in i:
            img, conv = convert(read(TRAIN_DIR + i))
            save(TRAIN_CACHE_DIR + i, img)
            train.append(conv)
    train_files = ls(TRAIN_CACHE_DIR)
    print('Done')

if FORCE_CONVERT or len(test) < 12500:
    sys.stdout.write('Process test data...')
    sys.stdout.flush()
    for i in os.listdir(TEST_DIR):
        if 'jpg' in i:
            img, conv = convert(read(TEST_DIR + i))
            save(TEST_CACHE_DIR + i, img)
            test.append(conv)
    test_files = ls(TEST_CACHE_DIR)
    print('Done')

train = np.array(train)
test = np.array(test)

print(f'Train shape: {train.shape}')
print(f'Train shape: {test.shape}')

labels = []

for i in train_files:
    if 'dog' in i:
        labels.append(0)
    else:
        labels.append(1)

labels = to_categorical(labels)

def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='same',
            activation='relu',
            kernel_initializer=trunc,
            bias_initializer=cnst,
            **kwargs
            )
def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
            units,
            activation='tanh',
            kernel_initializer=trunc,
            bias_initializer=cnst,
            **kwargs
            )

def AlexNet():
    model = Sequential()

    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(ROWS, COLS, 3)))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = AlexNet()
model.summary()

tensorBoard = TensorBoard(log_dir = 'log', histogram_freq=1, write_graph=True, write_grads=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
csvlogger = CSVLogger('log/epoch_log.csv')

history = model.fit(train, labels, epochs=2, batch_size=128, shuffle=True, validation_split=0.25, callbacks=[early_stopping, tensorBoard, csvlogger])

#predictions = model.predict(test, verbose=0)
