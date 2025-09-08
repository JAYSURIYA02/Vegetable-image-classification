import joblib
import cv2

classifier,vege_name= joblib.load('svm_vegetable_model.pkl')
print("Model loaded successfully.")

image_path = r'test\Papaya\1202.jpg'

img = cv2.imread(image_path)
resized_img = cv2.resize(img, (15, 15))
flattened_img = resized_img.flatten().reshape(1, -1)
prediction = classifier.predict(flattened_img)

print("Predicted class:", vege_name[prediction[0]])