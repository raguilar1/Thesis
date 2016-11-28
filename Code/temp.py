import numpy as np;
import matplotlib.pyplot as plot;
import PIL;
import readroi;
import CellDetect as cd
from os import listdir
import helperRead
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
import imageHelper

im = PIL.Image.open('test.tif');
imarray = np.array(im, 'float16');
print imarray.shape[1]
print im.size
print imarray.max()
imarray *= 1/(imarray.max())
print imarray.max()
rois = readroi.read_roi_zip('test.zip')
roiImage = imageHelper.getRoiImage(rois, imarray.shape);
filepath = 'C:\\Users\\rob\\Documents\\Thesis\\tp_approved'
trainImages = helperRead.getImages(filepath)
trainRois = helperRead.getRois(filepath)
Data = imageHelper.getTrainBatchFromImages(trainImages, trainRois, 100, 10, True)
Input_train = np.array(Data[0][0:400])
#Input_train = Input_train.reshape(400, 1, 40, 40)
Label_train = np.array(Data[1][0:400])
#Label_train = Label_train.reshape(400, 2, 1)
Input_test = np.array(Data[0][400:500])
#Input_test = Input_test.reshape(100, 1, 40, 40)
Label_test = np.array(Data[1][400:500])
#Label_test = Label_test.reshape(100, 2, 1)
#model = Sequential()
#model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(1,40,40)))
#model.add(Activation('relu'))
#model.add(Convolution2D(16, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (1,2,2)))
#model.add(Convolution2D(16, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size = (1,2,2)))
#model.add(Flatten())
#model.add(Dense(20))
#model.add(Activation('relu'))
#model.add(Dense(2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy',
#              optimizer='adadelta',
#              metrics=['accuracy'])
#model.fit(Input_train, Label_train, batch_size = 100, nb_epoch = 5, verbose =1, validation_data=(Input_test, Label_test))

#plot.imshow(roiImage)
#plot.imshow(imarray)
