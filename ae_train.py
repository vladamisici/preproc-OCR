import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

path = 'data/'

# Store image names in list for later use
train_img = sorted(os.listdir(path + 'train/'))
train_cleaned_img = sorted(os.listdir(path + 'train_cleaned/'))
test_img = sorted(os.listdir(path + 'test/'))
test_invoice_img = sorted(os.listdir(path + 'test_invoice/'))

# Define image dimensions
IMG_WIDTH = 540
IMG_HEIGHT = 420

# Prepare function to process images
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (IMG_HEIGHT, IMG_WIDTH, 1))
    
    return img

# Preprocess images
train = []
train_cleaned = []
test = []
test_invoice = []

for f in train_img:
    train.append(process_image(path + 'train/' + f))

for f in train_cleaned_img:
    train_cleaned.append(process_image(path + 'train_cleaned/' + f))
    
for f in test_img:
    test.append(process_image(path + 'test/' + f))
    
for f in test_invoice_img:
    test_invoice.append(process_image(path + 'test_invoice/' + f))

# Convert lists to numpy arrays
X_train = np.asarray(train)
Y_train = np.asarray(train_cleaned)
X_test = np.asarray(test)
X_test_invoice = np.asarray(test_invoice)

# Split data into training and validation sets
X_train_final, X_val, Y_train_final, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)

# Create the model
def model():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Dropout(0.5)(x)
    
    # decoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(x)
    
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    
    return model

# Create and compile the model
model = model()
model.summary()

# Train the model
BATCH_SIZE = 12
EPOCHS = 80

history = model.fit(
    X_train_final, Y_train_final,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val)
)

# Create directory for checkpoints
os.makedirs('./checkpoints/autoencoders', exist_ok=True)

# Save the model weights
model.save_weights('./checkpoints/autoencoders/checkpoint_kaggle_80eps.weights.h5')
model.save('./checkpoints/autoencoders/model_kaggle_80eps.keras')

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
