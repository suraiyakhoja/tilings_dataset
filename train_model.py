import tensorflow as tf from tensorflow 
import keras from tensorflow.keras 
import layers 
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score 
import os 
# ===================== 
# CONFIG 
# ===================== 

img_size = 224 
batch_size = 32 
data_dir = "tilings_dataset" 
seed = 42 

# ===================== 
# LOAD TRAIN SET # 
# =====================

train_ds = tf.keras.preprocessing.image_dataset_from_directory( 
    data_dir, 
    validation_split=0.2, 
    subset="training", 
    seed=seed, 
    image_size=(img_size, img_size), 
    batch_size=batch_size 
) 

# ===================== 
# LOAD FULL VALIDATION (will split into val + test) 
# ===================== 
val_full = tf.keras.preprocessing.image_dataset_from_directory( 
    data_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=seed, 
    image_size=(img_size, img_size), 
    batch_size=batch_size, 
    shuffle=True 
)

class_names = train_ds.class_names 
print("Classes:", class_names) 

# ===================== 
# SPLIT VALIDATION → VAL + TEST 
# ===================== 
val_batches = tf.data.experimental.cardinality(val_full) 
test_ds = val_full.take(val_batches // 2) 
val_ds = val_full.skip(val_batches // 2) 

# ===================== 
# PREFETCH
# =====================
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.prefetch(AUTOTUNE) 
val_ds = val_ds.prefetch(AUTOTUNE) 
test_ds = test_ds.prefetch(AUTOTUNE) 


# ===================== 
# DATA AUGMENTATION 
# ===================== 
augment = keras.Sequential([ 
    layers.RandomFlip("horizontal_and_vertical"), 
    layers.RandomRotation(0.25), 
    layers.RandomZoom(0.2), 
    layers.RandomContrast(0.2), 
]) 

train_ds = train_ds.map(lambda x, y: (augment(x), y)) 

# ===================== 
# BASE MODEL 
# =====================
base_model = keras.applications.MobileNetV2( 
    input_shape=(img_size, img_size, 3), 
    include_top=False, 
    weights="imagenet" 
) 

base_model.trainable = False 

# ===================== 
# CLASSIFIER HEAD 
# ===================== 
inputs = keras.Input(shape=(img_size, img_size, 3)) 
x = keras.applications.mobilenet_v2.preprocess_input(inputs) 
x = augment(x) 
x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dropout(0.3)(x) 
x = layers.Dense(128, activation="relu")(x) 
x = layers.Dropout(0.2)(x) 
outputs = layers.Dense(len(class_names), activation="softmax")(x) 

model = keras.Model(inputs, outputs)

model.compile( 
    optimizer=keras.optimizers.Adam(1e-3), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"] 
) 

model.summary() 

# ===================== 
# TRAIN HEAD 
# ===================== 
model.fit( 
    train_ds, 
    validation_data=val_ds, 
    epochs=10 
) 

# ===================== 
# FINE-TUNE LAST 30 LAYERS 
# =====================
base_model.trainable = True 
for layer in base_model.layers[:-30]: 
    layer.trainable = False 

model.compile( 
    optimizer=keras.optimizers.Adam(1e-5), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"] 
) 

early = keras.callbacks.EarlyStopping( 
    patience=5, 
    restore_best_weights=True 
) 

model.fit( 
    train_ds, 
    validation_data=val_ds, 
    epochs=30, 
    callbacks=[early] 
)

# ===================== 
# TEST SET EVALUATION 
# ===================== 
print("\n📊 Evaluating on test set...") 
y_true = [] 
y_pred = [] 

for images, labels in test_ds: 
    preds = model.predict(images) 
    y_true.extend(labels.numpy()) 
    y_pred.extend(np.argmax(preds, axis=1)) 

y_true = np.array(y_true) 
y_pred = np.array(y_pred) 

print("\n✅ Test Set Performance") 
print("Accuracy :", accuracy_score(y_true, y_pred)) 
print("Precision:", precision_score(y_true, y_pred, average="weighted")) 
print("Recall :", recall_score(y_true, y_pred, average="weighted")) 


# =====================
# SAVE MODEL 
# ===================== 
model.save("tiling_classifier.keras") 

print("\nModel saved as tiling_classifier.keras")
