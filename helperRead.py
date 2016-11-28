# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:40:42 2016

@author: rob
"""
from os import listdir
import readroi
import PIL
import numpy as np
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