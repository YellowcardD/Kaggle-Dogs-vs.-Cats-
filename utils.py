import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras.backend as K
import keras
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import re
from data_augmentation import *
import pandas as pd
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


# we will crop and resize input images to IMG_SIZE x IMG_SIZE
IMG_SIZE = 250
raw_train_images_path = r'dogs_and_cats_dataset/train/'
raw_test_images_path = r'dogs_and_cats_dataset/test/'
saved_train_images_path = r'dogs_and_cats_dataset/train_preprocessed_v1/'


def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height
    For cropping use numpy slicing
    """

    h, w = img.shape[0], img.shape[1]
    if w > h:
        cropped_img = img[:, (w - h) // 2 : (w - h) // 2 + h, :]
    else:
        cropped_img = img[(h - w) // 2 : (h - w) // 2 + w, :, :]

    return cropped_img

def preprocess_image_for_training(img):
    cropped_image = image_center_crop(img)
    image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    return image

def perpare_train_images_from_file(path):

    for i in range(0, 12500):
        name = 'cat.%d.jpg' %i
        img = cv2.imread(path + name)
        img = preprocess_image_for_training(img)
        cv2.imwrite(saved_train_images_path + name, img)

    for i in range(0, 12500):
        name = 'dog.%d.jpg' %i
        img = cv2.imread(path + name)
        img = preprocess_image_for_training(img)
        cv2.imwrite(saved_train_images_path + name, img)

def data_augmentation_v1(path):

    cnt = 0
    for i in range(0, 12500):
        base_name = 'cat.{}.jpg'
        img = cv2.imread(path + base_name.format(i))
        # print(path + base_name.format(i))
        # print(img)

        img1 = random_crop(img, 0.8, 0.1)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img1)
        cnt = cnt + 1

        img2 = rotate_image(img, 90, True)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img2)
        cnt = cnt + 1

        img3 = random_hsv_transform(img, 20, 0.3, 0.3)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img3)
        cnt = cnt + 1

        rand = np.random.randint(1, 4)
        if rand == 1:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img1, 1))
            cnt = cnt + 1
        if rand == 2:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img2, 1))
            cnt = cnt + 1
        if rand == 3:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img3, 1))
            cnt = cnt + 1

    for i in range(0, 12500):
        base_name = 'dog.{}.jpg'
        img = cv2.imread(path + base_name.format(i))

        img1 = random_crop(img, 0.8, 0.1)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img1)
        cnt = cnt + 1

        img2 = rotate_image(img, 90, True)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img2)
        cnt = cnt + 1

        img3 = random_hsv_transform(img, 40, 0.3, 0.3)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img3)
        cnt = cnt + 1

        rand = np.random.randint(1, 4)
        if rand == 1:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img1, 1))
            cnt = cnt + 1
        if rand == 2:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img2, 1))
            cnt = cnt + 1
        if rand == 3:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img3, 1))
            cnt = cnt + 1

def data_augmentation_v2(path):
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                 height_shift_range=0.2, channel_shift_range=15, horizontal_flip=True)
    gendata = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=saved_train_images_path, target_size=(IMG_SIZE, IMG_SIZE), save_format='jpg')
    for i in range(100000):
        gendata.next()

def data_augmentation_v3(path):

    cnt = 0
    for i in range(0, 12500):
        base_name = 'cat.{}.jpg'
        img = cv2.imread(path + base_name.format(i))
        # print(path + base_name.format(i))
        # print(img)

        img1 = random_crop(img, 0.8, 0.1)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img1)
        cnt = cnt + 1

        img2 = rotate_image(img, 90, True)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img2)
        cnt = cnt + 1

        img3 = random_hsv_transform(img, 20, 0.3, 0.3)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img3)
        cnt = cnt + 1

        rand = np.random.randint(1, 4)
        if rand == 1:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img1, 1))
            cnt = cnt + 1
        if rand == 2:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img2, 1))
            cnt = cnt + 1
        if rand == 3:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img3, 1))
            cnt = cnt + 1

    for i in range(0, 12500):
        base_name = 'dog.{}.jpg'
        img = cv2.imread(path + base_name.format(i))

        img1 = random_crop(img, 0.8, 0.1)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img1)
        cnt = cnt + 1

        img2 = rotate_image(img, 90, True)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img2)
        cnt = cnt + 1

        img3 = random_hsv_transform(img, 40, 0.3, 0.3)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(saved_train_images_path + base_name.format(cnt), img3)
        cnt = cnt + 1

        rand = np.random.randint(1, 4)
        if rand == 1:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img1, 1))
            cnt = cnt + 1
        if rand == 2:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img2, 1))
            cnt = cnt + 1
        if rand == 3:
            cv2.imwrite(saved_train_images_path + base_name.format(cnt), cv2.flip(img3, 1))
            cnt = cnt + 1

def data_augmentation_v4(path):
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                                 channel_shift_range=15, horizontal_flip=True)
    gendata = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=saved_train_images_path,
                                          target_size=(IMG_SIZE, IMG_SIZE), save_format='jpg')
    for i in range(100000):
        gendata.next()

def ResNet50(IMG_SIZE):

    model = keras.applications.ResNet50(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    output = GlobalAveragePooling2D()(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=model.inputs, outputs=output)

    return model

def VGG16(IMG_SIZE):

    model = keras.applications.VGG16(include_top=False,  weights='imagenet')
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    output = GlobalAveragePooling2D()(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=model.inputs, outputs=output)
    print(model.summary())

    return model

def InceptionResNetV2(IMG_SIZE):
    model = keras.applications.InceptionResNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    output = GlobalAveragePooling2D()(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=model.inputs, outputs=output)
    print(model.summary())

    return model

def Xception(IMG_SIZE):
    model = keras.applications.Xception(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    output = GlobalAveragePooling2D()(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=model.inputs, outputs=output)

    return model
    

