import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =========================
# PATHS
# =========================
BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pcb_defect_model_v3.h5")

# =========================
# IMAGE SETTINGS
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 30

# =========================
# DATA GENERATOR
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CLASS WEIGHTS (IMPORTANT)
# =========================
class_weight = {
    0: 1.5,  # Non-defective
    1: 1.0   # Defective
}

# =========================
# TRAIN
# =========================
model.fit(
    train_generator,
    epochs=EPOCHS,
    class_weight=class_weight
)

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_PATH)
print("✅ Model saved at:", MODEL_PATH)
