import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================
# PROJECT ROOT (run from PCB_Defect_Detection folder)
# ==================================================
BASE_DIR = os.getcwd()

# ==================================================
# MODEL (AFTER RETRAINING)
# ==================================================
MODEL_NAME = "pcb_defect_model_v3.h5"   # retrained model
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

# ==================================================
# TEST DATA PATH
# ==================================================
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

print("\n📂 Project root:", BASE_DIR)
print("📦 Using model:", MODEL_NAME)
print("📍 Model path:", MODEL_PATH)
print("🧪 Test folder:", TEST_DIR)

# ==================================================
# SAFETY CHECKS
# ==================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\n❌ Model file NOT found at:\n{MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"\n❌ Test folder NOT found at:\n{TEST_DIR}")

# ==================================================
# LOAD MODEL
# ==================================================
model = tf.keras.models.load_model(MODEL_PATH)
print("\n✅ Model loaded successfully")

# ==================================================
# TEST DATA GENERATOR
# ==================================================
IMG_SIZE = (224, 224)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# ==================================================
# PREDICTIONS
# ==================================================
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# ==================================================
# METRICS
# ==================================================
accuracy = accuracy_score(y_true, y_pred)

print("\n🎯 TEST ACCURACY:", round(accuracy * 100, 2), "%")

print("\n📊 CONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\n📄 CLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Non-Defective", "Defective"],
        zero_division=0   # avoids sklearn warnings
    )
)
