import os
import numpy as np
import pandas as pd
import csv

import cv2
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
from sklearn.model_selection import train_test_split


def get_all_filenames(dir):
    """
    Get get file names for the appointed direction
    """
    return [fn for fn in os.listdir(dir)]


all_fn = sorted(get_all_filenames('dogs_and_cats_dataset/train/cats'), key=lambda i: int(i.split('.')[1])) \
         + sorted(get_all_filenames('dogs_and_cats_dataset/train/dogs'), key=lambda i: int(i.split('.')[1]))
all_label = np.zeros((25000, ))
# rescale the label for 'imperfect' images
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


def get_array(path):
    """
    return an array from a csv file
    """
    data = pd.read_csv(path)
    data = data.iloc[:, 1].values
    data = np.array(data).reshape(-1, 1)
    return data

# get meta-features using the prediction from 20 models
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
np.save('X_train.npy', predict)
np.save('y_train.npy', tr_labels)

def create_model():
    """
    create a Logistic Regression model as meta-model
    """
    X_input = Input(shape=[20])
    X = Dense(1, activation='sigmoid')(X_input)
    model = Model(inputs=X_input, outputs=X)
    return model

# load data
X_train = np.load('stacking/X_train.npy')
y_train = np.load('stacking/y_train.npy')
X_val = np.load('stacking/X_val.npy')
y_val = np.load('stacking/y_val.npy')
X_test = np.load('stacking/X_test.npy')

# fit model using the defined model
batch_size = 32
sgd = keras.optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True, decay=0)
model = create_model()
model.compile(loss='binary_crossentropy', optimizer=sgd)
hist = model.fit(x=X_train, y=y_train, epochs=300, verbose=2,
                 validation_data=(X_val, y_val), shuffle=False)
print(hist.history)
model.save('stacking/model.hdf5')
model = load_model('stacking/model.hdf5')

# inference stage
predict = model.predict(X_test, batch_size=X_test.shape[0])
# clip the probability
res = np.clip(predict, 0.005, 0.995)
# write result to csv file
with open("stacking/final_submission.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id","label"])
    for i in range(12500):
        writer.writerow([i+1, np.squeeze(res[i])])
