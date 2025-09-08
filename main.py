import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path = r"Vegetable_images"
data =[]
labels = []

categories = os.listdir(path)
for category in categories:
  terms = os.path.join(path, category)
  vege_name = os.listdir(terms)
  for vege_ind,vege in enumerate(vege_name):
    inpath = os.path.join(terms, vege)
    for img in os.listdir(inpath):
      img_array = cv2.imread(os.path.join(inpath, img))
      resized_array = cv2.resize(img_array, (100, 100))
      flattened_array = resized_array.flatten()
      data.append(flattened_array)
      labels.append(vege_ind)





    
 