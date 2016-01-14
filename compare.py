import cv2
import numpy as np
import os
import sys
from path import path
import cPickle

dataPath="./Data"
desPath="./Desc"
hists=cPickle.load(open(dataPath + '/Hists','rb'))
des=cPickle.load(open(desPath + '/desc-moulinrouge6','rb'))

def distance (a, b):
	dist = sum(abs(a-b))
	return dist
