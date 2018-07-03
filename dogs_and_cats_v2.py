## augmentation v1 and ten-crop resolution=250, model=inception_v3

from utils import *
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import cv2
import csv
from keras.callbacks import ReduceLROnPlateau
import pandas as pd

batch_size = 64

def inception():

    model = keras.applications.InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    output = GlobalAveragePooling2D()(model.output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=model.inputs, outputs=output)

    return model

def get_all_filenames(dir):

    return [fn for fn in os.listdir(dir)]

all_fn = get_all_filenames('dogs_and_cats_dataset/train_preprocessed_v2')
all_fn = sorted(all_fn, key = lambda i:int(i.split('.')[1]))
print(all_fn[:10])
all_label = np.zeros((100000, ))
all_label[50000:] = 1
all_label[96924: 96924 + 4] = 0.001
all_label[67336: 67336 + 4] = 0.001
all_label[18752: 18752 + 4] = 0.5
all_label[44888: 44888 + 4] = 0.5
all_label[5800: 5800 + 4] = 0.5
all_label[8636: 8636 + 4] = 0.5
all_label[15288: 15288 + 4] = 0.5
all_label[16416: 16416 + 4] = 0.5
all_label[21420: 21420 + 4] = 0.5
all_label[28776: 28776 + 4] = 0.5
all_label[31680: 31680 + 4] = 0.5
all_label[37000: 37000 + 4] = 0.5
all_label[37776: 37776 + 4] = 0.5
all_label[39528: 39528 + 4] = 0.5
all_label[96152: 96152 + 4] = 0.5
all_label[84028: 84028 + 4] = 0.5
all_label[11756: 11756 + 4] = 0.9
all_label[12864: 12864 + 4] = 0.9
all_label[19332: 19332 + 4] = 0.9
all_label[31872: 31872 + 4] = 0.9
all_label[33880: 33880 + 4] = 0.9
all_label[90644: 90644 + 4] = 0.9
all_label[90760: 90760 + 4] = 0.9
all_label[94744: 94744 + 4] = 0.9
all_label[55232: 55232 + 4] = 0.9
all_label[57580: 57580 + 4] = 0.9
all_label[86752: 86752 + 4] = 0.9
all_label[21404: 21404 + 4] = 0.5
all_label[21672: 21672 + 4] = 0.5
all_label[36684: 36684 + 4] = 0.5
all_label[92988: 92988 + 4] = 0.5
all_label[60456: 60456 + 4] = 0.5
all_label[67468: 67468 + 4] = 0.5
all_label[84944: 84944 + 4] = 0.5
all_label[29508: 29508 + 4] = 0.5
all_label[99504: 99504 + 4] = 0.5
all_label[57092: 57092 + 4] = 0.5
all_label[42848: 42848 + 4] = 0.5
all_label[44736: 44736 + 4] = 0.5
all_label[30256: 30256 + 4] = 0.5
all_label[33824: 33824 + 4] = 0.5
all_label[90948: 90948 + 4] = 0.5
all_label[54172: 54172 + 4] = 0.5
all_label[54776: 54776 + 4] = 0.5
all_label[72416: 72416 + 4] = 0.5
all_label[88068: 88068 + 4] = 0.5
all_label[46260: 46260 + 4] = 0.5
all_label[93188: 93188 + 4] = 0.5
all_label[61508: 61508 + 4] = 0.5
all_label[85592: 85592 + 4] = 0.5

tr_files, te_files, tr_labels, te_labels = train_test_split(all_fn, all_label, test_size=0.2, random_state=42)
print(tr_files[:10])
print(tr_labels[:10])

def generator_with_label_from_file(filename, files, labels):
    label_by_fn = dict(zip(files, labels))
    for file in files:
        if file in os.listdir(filename):
            yield cv2.imread(filename + '/' + file), label_by_fn[file]

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


def train_generator(files, labels):
    while True:
        for batch in batch_generator(generator_with_label_from_file('dogs_and_cats_dataset/train_preprocessed_v2', files, labels), batch_size):
            # print(batch)
            batch_imgs = []
            batch_targets = []
            for img, target in batch:
                img = img.astype("float32")  # prepare for normalization
                img = keras.applications.inception_v3.preprocess_input(img)  # normalize for model
                batch_imgs.append(img)
                batch_targets.append(target)

            batch_imgs = np.stack(batch_imgs, axis=0)
            batch_targets = np.array(batch_targets).reshape((len(batch_targets), 1))
            yield batch_imgs, batch_targets


def generator_test_from_file(filename):
    for i in range(12500):
        yield cv2.imread(filename + '/{}.jpg'.format(i+1))


def test_generator():
    while True:
        for batch in batch_generator(generator_test_from_file('dogs_and_cats_dataset/test'), batch_size):
            batch_imgs = []
            for img in batch:
                img = image_center_crop(img)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32")  # prepare for normalization
                img = keras.applications.inception_v3.preprocess_input(img)  # normalize for model
                batch_imgs.append(img)

            batch_imgs = np.stack(batch_imgs, axis=0)
            yield batch_imgs

def get_ten_crop_imgs(img):
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

def ten_crop_test_generator():
    while True:
        for batch in batch_generator(generator_test_from_file('dogs_and_cats_dataset/test'), 1):
            ten_crop_imgs = get_ten_crop_imgs(batch[0])
            yield ten_crop_imgs



# for X in train_generator(tr_files, tr_labels):
#     print(X)

model = inception()
#print(model.summary())
#print(len(model.layers))  313
for layer in model.layers:
    layer.trainable = True
for layer in model.layers[:-100]:
    layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam')
model_filename = 'model/dogs_and_cats_{0:03d}.hdf5'
last_finished_epoch = None

from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 15
# model = load_model(model_filename.format(last_finished_epoch))

from keras_tqdm import TQDMCallback
from keras.callbacks import ModelCheckpoint

decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='max')
# model.fit_generator(
#     train_generator(tr_files, tr_labels),
#     steps_per_epoch = len(tr_files) // batch_size // 8,
#     epochs = 2*8,
#     validation_data = train_generator(te_files, te_labels),
#     validation_steps = len(te_files) //  batch_size // 4,
#     callbacks=[TQDMCallback(), ModelCheckpoint('model/v3_correct_label.hdf5', save_best_only=True), decay],
#     verbose=0,
#     initial_epoch=last_finished_epoch or 0
# )

model = load_model('model/submission_v6_aug1_ten-crop_resolution=250_model=inception_v3.hdf5')
# result = model.predict_generator(ten_crop_test_generator(), 12500 // 32 + 1)
result_temp = model.predict_generator(ten_crop_test_generator(), 12500)
print(result_temp.shape)
result_temp = np.squeeze(result_temp)
print(result_temp.shape)
result = np.zeros((12500, ))
for i in range(12500):
    result[i] = np.mean(result_temp[i*10: i*10 + 10])

with open("submission_v6_aug1_ten-crop_resolution=250_model=inception_v3.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id","label"])
    for i in range(12500):
        writer.writerow([i+1, np.squeeze(result[i])])

result = np.clip(result, 0.01, 0.99)
with open("submission_v6_aug1_ten-crop_resolution=250_model=inception_v3_clipped.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id","label"])
    for i in range(12500):
        writer.writerow([i+1, np.squeeze(result[i])])
