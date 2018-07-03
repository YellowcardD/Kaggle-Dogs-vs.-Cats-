import cv2
import keras
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import csv
import time
import pandas as pd

def get_all_filenames(dir):

    return [fn for fn in os.listdir(dir)]

all_fn = sorted(get_all_filenames('dogs_and_cats_dataset/train/cats'), key = lambda i:int(i.split('.')[1])) + sorted(get_all_filenames('dogs_and_cats_dataset/train/dogs'), key = lambda i:int(i.split('.')[1]))
all_label = np.zeros((25000, ))
all_label[12500:] = 1
all_label[24231] = 0.001
all_label[16834] = 0.001
all_label[4688] = 0.5
all_label[11222] = 0.5
all_label[1450] = 0.5
all_label[2159] = 0.5
all_label[3822] = 0.5
all_label[4104] = 0.5
all_label[5355] = 0.5
all_label[7194] = 0.5
all_label[7920] = 0.5
all_label[9250] = 0.5
all_label[9444] = 0.5
all_label[9882] = 0.5
all_label[24038] = 0.5
all_label[21007] = 0.5
all_label[2939] = 0.9
all_label[3216] = 0.9
all_label[4833] = 0.9
all_label[7968] = 0.9
all_label[8470] = 0.9
all_label[22661] = 0.9
all_label[22690] = 0.9
all_label[23686] = 0.9
all_label[13808] = 0.9
all_label[14395] = 0.9
all_label[21688] = 0.9
all_label[5351] = 0.5
all_label[5418] = 0.5
all_label[9171] = 0.5
all_label[23247] = 0.5
all_label[15114] = 0.5
all_label[16867] = 0.5
all_label[21236] = 0.5
all_label[7377] = 0.5
all_label[24876] = 0.5
all_label[14273] = 0.5
all_label[10712] = 0.5
all_label[11184] = 0.5
all_label[7564] = 0.5
all_label[8456] = 0.5
all_label[22737] = 0.5
all_label[13543] = 0.5
all_label[13694] = 0.5
all_label[18104] = 0.5
all_label[22017] = 0.5
all_label[11565] = 0.5
all_label[23297] = 0.5
all_label[15377] = 0.5
all_label[21398] = 0.5


tr_files, te_files, tr_labels, te_labels = train_test_split(all_fn, all_label, test_size=0.2, random_state=5)
# print(tr_files[:30])
# print(tr_labels[:30])

def batch_generator(items, batch_size):

    minibatch = []
    cnt = 0
    for item in items:
        minibatch.append(item)
        cnt = cnt + 1
        if cnt == batch_size:
            yield minibatch
            minibatch = []
            cnt = 0
    if cnt != 0:
        yield minibatch

def get_ten_crop_imgs(img, IMG_SIZE):
    batch_imgs = []
    img = cv2.resize(img, (IMG_SIZE + 40, IMG_SIZE + 40))
    #print(img.shape)
    batch_imgs.append(img[:IMG_SIZE, :IMG_SIZE, :])
    batch_imgs.append(img[:IMG_SIZE, 40:, :])
    batch_imgs.append(img[40:, :IMG_SIZE, :])
    batch_imgs.append(img[40:, 40:, :])
    batch_imgs.append(img[20:IMG_SIZE+20, 20:IMG_SIZE+20, :])
    for i in range(5):
        batch_imgs.append(cv2.flip(batch_imgs[i], 1))
    for i in range(10):
        batch_imgs[i] = batch_imgs[i].astype("float32")  # prepare for normalization
        batch_imgs[i] = keras.applications.inception_v3.preprocess_input(batch_imgs[i])  # normalize for model
    batch_imgs = np.stack(batch_imgs, axis=0)
    #print(batch_imgs.shape)
    return batch_imgs

def generator_test_from_file():
    for i in te_files:
        # print(i)
        yield cv2.imread('dogs_and_cats_dataset/train/' + i)


def ten_crop_test_generator(IMG_SIZE):
    while True:
        for batch in batch_generator(generator_test_from_file(), 1):
            ten_crop_imgs = get_ten_crop_imgs(batch[0], IMG_SIZE)
            yield ten_crop_imgs

def get_array(path):

    data = pd.read_csv(path)
    data = data.iloc[:, 1].values
    data = np.array(data).reshape(-1, 1)
    #print(data.shape)

    return data

predict = []
predict1 = get_array('stacking/aug1_ten-crop_resolution=250_model=InceptionV3.csv')
predict.append(predict1)
predict2 = get_array('stacking/aug1_ten-crop_resolution=299_model=InceptionV3.csv')
predict.append(predict2)
predict3 = get_array('stacking/aug2_ten-crop_resolution=250_model=InceptionV3.csv')
predict.append(predict3)
predict4 = get_array('stacking/aug2_ten-crop_resolution=299_model=InceptionV3.csv')
predict.append(predict4)
predict5 = get_array('stacking/aug1_ten-crop_resolution=250_model=Xception.csv')
predict.append(predict5)
predict6 = get_array('stacking/aug1_ten-crop_resolution=299_model=Xception.csv')
predict.append(predict6)
predict7 = get_array('stacking/aug2_ten-crop_resolution=250_model=Xception.csv')
predict.append(predict7)
predict8 = get_array('stacking/aug2_ten-crop_resolution=299_model=Xception.csv')
predict.append(predict8)
predict9 = get_array('stacking/aug1_ten-crop_resolution=250_model=ResNet50.csv')
predict.append(predict9)
predict10 = get_array('stacking/aug1_ten-crop_resolution=299_model=ResNet50.csv')
predict.append(predict10)
predict11 = get_array('stacking/aug2_ten-crop_resolution=250_model=ResNet50.csv')
predict.append(predict11)
predict12 = get_array('stacking/aug2_ten-crop_resolution=299_model=ResNet50.csv')
predict.append(predict12)
predict13 = get_array('stacking/aug1_ten-crop_resolution=250_model=InceptionResNetV2.csv')
predict.append(predict13)
predict14 = get_array('stacking/aug1_ten-crop_resolution=299_model=InceptionResNetV2.csv')
predict.append(predict14)
predict15 = get_array('stacking/aug2_ten-crop_resolution=250_model=InceptionResNetV2.csv')
predict.append(predict15)
predict16 = get_array('stacking/aug2_ten-crop_resolution=299_model=InceptionResNetV2.csv')
predict.append(predict16)
predict17 = get_array('stacking/aug1_ten-crop_resolution=250_model=VGG16.csv')
predict.append(predict17)
predict18 = get_array('stacking/aug1_ten-crop_resolution=299_model=VGG16.csv')
predict.append(predict18)
predict19 = get_array('stacking/aug2_ten-crop_resolution=250_model=VGG16.csv')
predict.append(predict19)
predict20 = get_array('stacking/aug2_ten-crop_resolution=299_model=VGG16.csv')
predict.append(predict20)

predict = np.hstack(predict)
predict = np.array(predict)
print(predict.shape)
np.save('X_train.npy', predict)
np.save('y_train.npy', tr_labels)
print(np.load('X_train.npy').shape)
print(np.load('y_train.npy').shape)

from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
def create_model():

    X_input = Input(shape=[20])
    X = Dense(1, activation='sigmoid')(X_input)

    model = Model(inputs=X_input, outputs=X)

    return model

X_train = np.load('stacking/X_train.npy')
y_train = np.load('stacking/y_train.npy')
X_val = np.load('stacking/X_val.npy')
y_val = np.load('stacking/y_val.npy')
X_test = np.load('stacking/X_test.npy')
print(X_train.shape, X_val.shape, X_test.shape)
batch_size = 32
sgd = keras.optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True, decay=0)
model = create_model()
model.compile(loss='binary_crossentropy', optimizer=sgd)
hist = model.fit(x=X_train, y=y_train, epochs=300, verbose=2,
          validation_data=(X_val, y_val), shuffle=False)
print(hist.history)
model.save('stacking/model.hdf5')
model = load_model('stacking/model.hdf5')

predict = model.predict(X_test, batch_size=X_test.shape[0])
print(predict.shape)

res = np.clip(predict, 0.005, 0.995)
with open("stacking/final_submission.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id","label"])
    for i in range(12500):
        writer.writerow([i+1, np.squeeze(res[i])])
