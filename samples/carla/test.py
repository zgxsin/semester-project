import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join


mypath = "/Users/zhou/Downloads/MyDocuments/Imaga_DataSet_Training/0_100/Dynamic/dataset/train_mask/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


print(onlyfiles.shape)