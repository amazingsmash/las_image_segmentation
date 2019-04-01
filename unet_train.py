import os
from idlelib.pyparse import trans

import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import skimage.io as io
import skimage.transform as trans
from matplotlib import pyplot as plt



def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def readImagesFromFiles(path, folder_x, folder_y, target_size=(256, 256)):
    path_x = path + "/" + folder_x
    files = [path_x + "/" + f for f in os.listdir(path_x) if f.endswith(".png")]

    x = np.zeros((len(files), target_size[0], target_size[1], 1), dtype=float)
    print(x.shape)
    for i, file in enumerate(files):
        print(file)
        img = io.imread(file, as_gray=True)
        img = img / 255
        img = trans.resize(img, (*target_size, 1))
        x[i] = img

        # print(img.shape)
        #plt.imshow(img[:,:,0], interpolation='nearest')
        #plt.show()

    path_y = path + "/" + folder_y
    files = [path_y + "/" + f for f in os.listdir(path_y) if f.endswith(".png")]
    y = np.zeros((len(files), target_size[0], target_size[1], 1), dtype=float)
    for i, file in enumerate(files):
        print(file)
        img = io.imread(file, as_gray=True)
        img = img > 0
        img = trans.resize(img, (*target_size, 1))
        y[i] = img

        # print(img.shape)
        plt.imshow(img[:,:,0], interpolation='nearest')
        plt.show()

    return x, y


def readImagesFromFiles(path, folder_x, target_size=(256, 256)):
    path_x = path + "/" + folder_x
    files = [path_x + "/" + f for f in os.listdir(path_x) if f.endswith(".png")]

    x = np.zeros((len(files), target_size[0], target_size[1], 1), dtype=float)
    print(x.shape)
    for i, file in enumerate(files):
        print(file)
        img = io.imread(file, as_gray=True)
        img = img / 255
        img = trans.resize(img, (*target_size, 1))
        x[i] = img

        # print(img.shape)
        #plt.imshow(img[:,:,0], interpolation='nearest')
        #plt.show()

    return x



def trainModel():
    x, y = readImagesFromFiles('upperView', 'image', 'label')
    model = unet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(x, y, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])


def testModel():
    x = readImagesFromFiles('upperView', 'test', target_size=(256, 256))
    model = unet()
    p = model.predict(x)

#model = trainModel()
testModel()