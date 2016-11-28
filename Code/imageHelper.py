# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:40:42 2016

@author: rob
"""
from os import listdir
import readroi
import PIL
import numpy as np
import random as rnd
#filepath = 'C:\\Users\\rob\\Documents\\Thesis\\tp_approved'

def getRois(filepath):
    roiFileNames = [f for f in listdir(filepath) if '.zip' in f ]
    rois = [];    
    for roiFile in roiFileNames:
        rois.append(readroi.read_roi_zip(filepath + '\\' + roiFile))
    return rois;
    
def getImages(filepath, datatype = 'float16'):
    imageFileNames = [f for f in listdir(filepath) if '.tif' in f ]
    images = [];
    for imageFile in imageFileNames:
        im = PIL.Image.open(filepath + '\\'+ imageFile);
        imarray = np.array(im, datatype);
        images.append(imarray)
    return images;

def getRoiImage(roiList, shape, datatype = 'float16', centered = True, radius = 5):
    roiImage = np.empty(shape, datatype)
    if not centered:
        for roi in roiList:
            for pixel in roi:
                roiImage[pixel[0], pixel[1]] = 1;
    else:
        for roi in roiList:
            xAverage = 0;
            yAverage = 0;
            numPixels = 0;
            for pixel in roi:
                xAverage += pixel[0];
                yAverage += pixel[1];
                numPixels += 1;
            xAverage = round(xAverage/ numPixels);
            yAverage = round(yAverage/ numPixels);
            for i in list(range( int(xAverage -radius),int(xAverage + radius))):
                for y in list(range(int(yAverage -radius), int(yAverage + radius))):
                    roiImage[i,y] = 1;
    return roiImage;
    
def getPatchFromImage(image, center, radius):
    return image[center[0]-radius : center[0]+radius, center[1]-radius : center[1]+radius];

def getTrainingBatchFromImage(image, rois, numPatches, patchRadius, balanced=True, startPatches =[], startLabels = []):
    patches = startPatches
    labels = startLabels
    roiImage = getRoiImage(rois, image.shape)
    if not balanced:
        for i in list(range(0,numPatches)):
            a = rnd.randint(patchRadius+1, image.shape[0] -(patchRadius +1))
            b = rnd.randint(patchRadius+1, image.shape[1] -(patchRadius +1))
            patches.append(getPatchFromImage(image, [a,b], patchRadius))
            label = roiImage[a,b]
            if label == 0:
                labels.append([1,0])
            else:
                labels.append([0,1])
    else:
        for i in list(range(0,int(round(numPatches/2)))):
            a = rnd.randint(patchRadius+1, image.shape[0] -(patchRadius +1))
            b = rnd.randint(patchRadius+1, image.shape[1] -(patchRadius +1))
            patches.append(getPatchFromImage(image, [a,b], patchRadius))
            label = roiImage[a,b]
            if label == 0:
                labels.append(np.array([1,0]))
            else:
                labels.append(np.array([0,1]))
        for i in list((range(int(round(numPatches/2)), numPatches))):
            roi = rois[rnd.randint(0, len(rois)-1)]
            xAverage = 0;
            yAverage = 0;
            numPixels = 0;
            for pixel in roi:
                xAverage += pixel[0];
                yAverage += pixel[1];
                numPixels += 1;
            xAverage = round(xAverage/ numPixels);
            yAverage = round(yAverage/ numPixels);
            rad = 2
            a = rnd.randint(xAverage - rad, xAverage+rad)
            b = rnd.randint(yAverage - rad, yAverage+rad)
            patches.append(getPatchFromImage(image, [a,b], patchRadius))
            labels.append(np.array([0,1]))
            
    return (patches, labels)
    
def getTrainBatchFromImages(images, roiLists, numPatchesPerImage, patchRadius, balanced=True):
    count = 0
    patches = [];
    labels = [];
    for image in images:
        temp = getTrainingBatchFromImage(image, roiLists[count], numPatchesPerImage, patchRadius, balanced, startPatches =patches, startLabels = labels)
        patches = temp[0]
        labels = temp[1]
        count = count + 1;
    return (patches, labels) 
        
        
    