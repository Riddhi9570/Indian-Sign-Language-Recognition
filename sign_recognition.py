import numpy as np
import cv2
import keras
import tensorflow as tf
from keras.preprocessing.image import img_to_array

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

model = keras.models.load_model("my_model.h5")
cam = cv2.VideoCapture(0)

map_characters = {0: '--', 1: 'Zero', 2: 'One', 3: 'Two', 4: 'Three', 5: 'Four', 6: 'Five', 7: 'Six', 8: 'Seven', 9: 'Eight',
                  10: 'Nine', 11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K',
                  22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W',
                  34: 'X', 35: 'Y', 36: 'Z'}
class_labels = list(map_characters.values())

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    roi = frame[10:300, 320:620]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("BW", final_image)
    final_image = cv2.resize(final_image, (64, 64))

    final_image = img_to_array(final_image)
    tp = final_image.reshape(1, 64, 64, 1)
    pred = model.predict(tp)
    print(class_labels[np.argmax(pred)])
    cv2.putText(frame, class_labels[np.argmax(pred)], (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
