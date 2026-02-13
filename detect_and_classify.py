import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


model = tf.keras.models.load_model("pcb_defect_model.h5")


class_names = ['Missing_hole', 'Mouse_bite', 'Open_circuit',
               'Short', 'Spur', 'Spurious_copper']

IMG_SIZE = 128


template_path = "template.jpg"  
test_path = "test.jpg"           

template = cv2.imread(template_path)
test = cv2.imread(test_path)


test = cv2.resize(test, (template.shape[1], template.shape[0]))


template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)


diff = cv2.absdiff(template_gray, test_gray)


_, thresh = cv2.threshold(diff, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)


kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.dilate(thresh, kernel, iterations=1)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

output_image = test.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

   
    if w * h < 200:
        continue

    roi = test[y:y+h, x:x+w]

    if roi.size == 0:
        continue

    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_array = np.expand_dims(roi_resized, axis=0)
    roi_array = preprocess_input(roi_array)

    prediction = model.predict(roi_array)
    class_index = np.argmax(prediction)
    label = class_names[class_index]
    confidence = np.max(prediction)

    cv2.rectangle(output_image, (x, y), (x+w, y+h),
                  (0, 255, 0), 2)

  
    text = f"{label} ({confidence:.2f})"
    cv2.putText(output_image, text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

cv2.imwrite("output_result.jpg", output_image)

print("Detection and classification completed.")
