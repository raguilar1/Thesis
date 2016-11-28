# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 22:25:55 2016

@author: rob
"""

def getPatchesFromImage(image, n):
    patches = []
    for i in list(range(0, image.shape[0]-n)):
        for j in list(range(0, image.shape[1]-n)):
            patches.append(image[i:i+n, j:j+n]);
    return patches;
    
#def getImageFromPatches(patches, shape):
    
