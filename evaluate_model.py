import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report


model = tf.keras.models.load_model("pcb_defect_model.h5")

print("\n✅ Model Loaded Successfully\n")
print(model.summary())


IMG_SIZE = 128
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input  
)

test_generator = test_datagen.flow_from_directory(
    "roi_split/test",      
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False          
)

loss, accuracy = model.evaluate(test_generator)

print("\n📊 Test Loss:", loss)
print("📊 Test Accuracy:", accuracy)


predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\n✅ Confusion Matrix saved as confusion_matrix.png")


report = classification_report(
    y_true,
    y_pred,
    target_names=test_generator.class_indices.keys()
)

print("\n📄 Classification Report:\n")
print(report)


with open("evaluation_report.txt", "w") as f:
    f.write("PCB Defect Detection Model Evaluation\n\n")
    f.write("Test Loss: " + str(loss) + "\n")
    f.write("Test Accuracy: " + str(accuracy) + "\n\n")
    f.write("Classification Report:\n\n")
    f.write(report)

print("✅ evaluation_report.txt saved successfully")
