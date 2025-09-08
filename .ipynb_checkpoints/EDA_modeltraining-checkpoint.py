import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib 


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
      resized_array = cv2.resize(img_array, (15, 15))
      flattened_array = resized_array.flatten()
      data.append(flattened_array)
      labels.append(vege_ind)

x = np.array(data)
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y,shuffle=True)
"""
Classifier =SVC(C=1, gamma=0.01,verbose=3)
Classifier.fit(x_train, y_train)
"""
Classifier = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)
Classifier.fit(x_train, y_train)

y_pred = Classifier.predict(x_test)

score = accuracy_score(y_test, y_pred)
print("Accuracy: ", score * 100)

# Save the trained model
joblib.dump((Classifier,vege_name),'svm_vegetable_model.pkl')





    
 