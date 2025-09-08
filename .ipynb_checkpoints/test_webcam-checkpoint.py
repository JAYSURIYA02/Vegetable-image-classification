import joblib
import cv2
import numpy as np

classifier,vege_name= joblib.load('svm_vegetable_model.pkl')
print("Model loaded successfully.")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (15, 15))
    flattened_frame = resized_frame.flatten().reshape(1, -1)
    prediction = classifier.predict(flattened_frame)


    cv2.putText(frame, f'Predicted class: {vege_name[prediction[0]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



