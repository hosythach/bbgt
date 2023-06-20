import tensorflow as tf

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}

from tensorflow.keras.models import Model
import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
saved_model = tf.keras.models.load_model("nhandienbienbaogiaothongver9_adam_optim.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 640) # Chiều rộng cửa sổ
cap.set(4, 480) # Chiều dài cửa sổ
cap.set(10, 180) # Độ sáng

while True:
    # Đọc ảnh từ Webcame
    success,imgOrignal = cap.read()
    
    # Xử lý ảnh
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))    
    img = img / 255.0
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 3)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    pred = np.argmax(saved_model.predict(img),axis=1)
    if np.amax(saved_model.predict(img)) > 0.9:
        cv2.putText(imgOrignal, classNames[pred[0]], (120, 35),font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(np.amax(saved_model.predict(img)) * 100, 2)) + " %", (180, 75),font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("KQ", imgOrignal)
    else: cv2.putText(imgOrignal, str("unrecognizable"), (120, 35),font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
