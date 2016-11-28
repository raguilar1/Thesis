# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 00:28:38 2016

@author: rob
"""

import numpy as np
import imageHelper
from scipy import ndimage
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

model = Sequential()

def initializeModel(patchRadius):
    patchSize = patchRadius *2
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(1,patchSize,patchSize)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (1,2,2)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (1,2,2)))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


def trainModel(inputImages, rois, patchesPerImagePerBatch, patchRadius, numBatches, numTrainImages):
    patchSize = patchRadius *2
    trainPatchesPerBatch = patchesPerImagePerBatch*numTrainImages
    testPatchesPerBatch = patchesPerImagePerBatch*(len(inputImages)-numTrainImages)
    for i in range(0, numBatches):
        Data = imageHelper.getTrainBatchFromImages(inputImages, rois, patchesPerImagePerBatch, patchRadius, True)
        Input_train = np.array(Data[0][0:trainPatchesPerBatch])
        Label_train = np.array(Data[1][0:trainPatchesPerBatch])
        Input_test = np.array(Data[0][trainPatchesPerBatch:trainPatchesPerBatch+testPatchesPerBatch])
        Label_test = np.array(Data[1][trainPatchesPerBatch:trainPatchesPerBatch+testPatchesPerBatch])
        #convert trai/test to correct shape
        Input_train = Input_train.reshape(trainPatchesPerBatch, 1, patchSize, patchSize)
        Label_train = Label_train.reshape(trainPatchesPerBatch, 2, 1)
        Input_test = Input_test.reshape(testPatchesPerBatch, 1, patchSize, patchSize)
        Label_test = Label_test.reshape(testPatchesPerBatch, 2, 1)
        #increase the nb_epoch if there is a lot of overhead for the model fit stuff
        model.fit(Input_train, Label_train, batch_size = 50, nb_epoch = 1, verbose =1, validation_data=(Input_test, Label_test))

def saveModel(filepath):
    model.save(filepath)

def loadPreTrainedModel(filepath):
    model = load_model(filepath)

#returns a list of numpy arrays of size n by 2
#aka returns a list of rois definied by the (x,y) points they contain
def getRoiFromNetwork(inputImage):
    probMap = getProbMapForImage(40, inputImage);
    pppm = postProcessProbMap(probMap, 'square_erosion');
    return getRoiFromPostProcessedProbMap(pppm);

def getProbMapForImage(n, image):
    return image;
    
def postProcessProbMap(probMap, algorithm = 'erosion'):
    threshold = .9;
    if (algorithm == 'erosion'):
         pppm = probMap > threshold
         pppm = ndimage.binary_erosion(pppm)
    elif (algorithm == 'square'):    
        pppm = probMap**2 > threshold
    elif (algorithm == 'square_erosion'):
        pppm = probMap**2 > threshold
        pppm = ndimage.binary_erosion(pppm)
    else:
        pppm = probMap > threshold
    return pppm;
    
def getRoiFromPostProcessedProbMap(pppm):
    radius = 5;
    sizeThreshold = 0;
    mask = pppm > 0;
    #finds Connected Components
    cellLabelledImage, numCells = ndimage.label(mask)
    rois = [];
    #places ROI on top of center of CC
    for i in range(1,numCells):
        xAverage = 0;
        yAverage = 0;
        numPixels = 0;
        for x in list(range(0, pppm.shape[0])):
            for y in list(range(0, pppm.shape[1])):
                if(cellLabelledImage[x,y] == i):
                    xAverage = xAverage + x;
                    yAverage = yAverage + y;
                    numPixels = numPixels + 1;
        if numPixels > sizeThreshold :
            xAverage = round(xAverage / numPixels);
            yAverage = round(yAverage / numPixels);
            roi = []
            for x in list(range( int(xAverage -radius),int(xAverage + radius))):
                for y in list(range(int(yAverage -radius), int(yAverage + radius))):
                    roi.append([x,y])
            rois.append(np.array(roi));
    return rois;