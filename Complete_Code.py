from __future__ import division
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
import tensorflow as tf
import os
import numpy as np
import random
import math
import scipy.ndimage.interpolation as inter


def CNN_Model():

    skeleton_joints = Input(name='joints', shape=(31, 25, 3))
    skeleton_movement = Input(name='joints_diff', shape=(31, 25, 3))

    ### Joint Branch ####
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(skeleton_joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Permute((1, 3, 2))(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    ### Joint Branch End ###

    ### Temporal Branch ###
    x_d = Conv2D(filters=32, kernel_size=(1, 1),
                 padding='same')(skeleton_movement)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Permute((1, 3, 2))(x_d)

    x_d = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    ### Temporal Branch End ###

    x = concatenate([x, x_d], axis=-1)

    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(93, activation='softmax')(x)

    model = Model([skeleton_joints, skeleton_movement], x)

    return model


def missing_skele():
    f = open('data/ntu_rgb120_missing.txt', 'r')
    bad_files = []
    for line in f:
        bad_files.append(line.strip()+'.skeleton')
    f.close()
    return bad_files


def zoom(p):
    l = p.shape[0]
    p_new = np.empty([32, 25, 3])
    for m in range(25):
        for n in range(3):
            p_new[:, m, n] = inter.zoom(p[:, m, n], 32/l)[:32]
    return p_new


def normal(p):
    l = len(p)
    while l < 300:
        change = 300 - l
        p = np.append(p[:, :, :], p[:change, :, :], axis=0)
        l = len(p)
    return p


def data_prep():

    global npy_path
    global data_filename
    global train_filename
    global train_labels
    global test_filename
    global test_labels
    data_label = []

    # reading all filenames into list
    npy_path = '/home/mokeaveli/TYP/TYP/data/npy/'
    data_filename = os.listdir(npy_path)

    for record in data_filename:
        # reading the corresponding lables into a list
        data_label.append(int(record[17:20]))

    ### Create random allocation for training and testing data sets ###
    data = list(zip(data_filename, data_label))
    random.shuffle(data)
    data_filename, data_label = zip(*data)

    train_mult = round(random.uniform(0.6, 0.8), 2)
    total_records = len(data_filename)
    train_num = math.ceil(total_records * train_mult)

    train_filename = data_filename[:train_num]
    train_labels = data_label[:train_num]
    test_filename = data_filename[train_num:]
    test_labels = data_label[train_num:]


def get_data():
    train_body_data = []
    train_body_diff = []
    test_body_data = []
    test_body_diff = []

    bad_files = missing_skele()

    for file in data_filename:

        if file in bad_files:
            continue

        skele_data = np.load(npy_path + file, allow_pickle=True).item()
        person = skele_data.get('skel_body0')
        frames = len(skele_data.get('skel_body0'))  # total frames

        body_diff = []
        body_pos = []

        # sample the range from crop size of [32,frames]
        if frames > 32:
            ratio = np.random.uniform(1, frames/32)
            l = int(32*ratio)
            start = random.sample(range(frames-l), 1)[0]
            end = start+l
            person = person[start:end, :, :]
            person = zoom(person)
        elif frames < 32:
            person = zoom(person)

        body_pos = person[1:, :, :]
        body_diff = person[1:, :, :]-person[:-1, :, :]

        if file in train_filename:
            train_body_data.append(body_pos)
            train_body_diff.append(body_diff)

        if file in test_filename:
            test_body_data.append(body_pos)
            test_body_diff.append(body_diff)

    train_body_data = np.array(train_body_data)
    train_body_diff = np.array(train_body_diff)

    test_body_data = np.array(test_body_data)
    test_body_diff = np.array(test_body_diff)

    train_labels_np = np.array(train_labels)
    test_labels_np = np.array(test_labels)

    return [[train_body_data, train_body_diff, train_labels_np], [test_body_data, test_body_diff, test_labels_np]]


def main():
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model = CNN_Model()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    data_prep()
    # fit model
    [train, test] = get_data()
    history = model.fit([train[0], train[1]], train[2], epochs=10000,
                        verbose=True, shuffle=True)

    # evaluate the model
    pred = model.predict([test[0], test[1]])
    predicted_true = 0
    predicted_labels = np.argmax(pred, axis=1)
    ground_truth_labels = test[2]

    for i in range(len(predicted_labels)):
        if(predicted_labels[i] == ground_truth_labels[i]):
            predicted_true += 1
    print(predicted_true/len(predicted_labels)*100)


if __name__ == "__main__":
    main()
