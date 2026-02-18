import cv2
import os

# Correct dataset path
dataset_path = "PCB_USED"

# List files/folders inside dataset
files = os.listdir(dataset_path)
print("Contents of PCB_USED folder:", files)

# Pick the first image file automatically
for file in files:
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_path, file)
        break
else:
    print("No image found in PCB_USED folder.")
    exit()

image = cv2.imread(img_path)

if image is None:
    print("Image not loaded.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("PCB Image", image)
    cv2.imshow("Grayscale PCB Image", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
