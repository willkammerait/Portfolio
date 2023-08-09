#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# Load the data
train_data = pd.read_excel("/Users/wkammerait/Desktop/ML Data Sets/data/Train_data_label.xlsx")
test_data = pd.read_excel("/Users/wkammerait/Desktop/ML Data Sets/data/Test_data_label.xlsx")
folder_path = "/Users/wkammerait/Desktop/ML Data Sets/data/"
train_data['Path'] = folder_path + train_data['Path']
test_data['Path'] = folder_path + test_data['Path']


# In[2]:


# Load and resize images
def load_and_resize_images(data):
    images = []
    for path in data['Path']:
        img = cv2.imread(path)
        img = cv2.resize(img, (30, 30))
        images.append(img)
    return np.array(images)

train_images = load_and_resize_images(train_data)
test_images = load_and_resize_images(test_data)

train_images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images])
test_images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in test_images])

# Normalize the pixel values to be in the range [0, 1]
train_images = train_images.astype('float32') / 255.0
train_images_gray = train_images_gray.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
test_images_gray = test_images_gray.astype('float32') / 255.0

# Convert the labels to categorical format
y_train = np.array(train_data['ClassId'])
y_test = np.array(test_data['ClassId'])
y_train_cat = to_categorical(y_train, num_classes=43)
y_test_cat = to_categorical(y_test, num_classes=43)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prior to oversampling
class_counts = pd.Series(y_train).value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class distribution before oversampling")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()


# In[4]:


# Handle class imbalance with oversampling
counter = Counter(y_train)
mean_count = int(np.mean([count for label, count in counter.items()]))
strategy = {label: max(count, mean_count) for label, count in counter.items()}
oversample = RandomOverSampler(sampling_strategy=strategy)
train_images_res, y_train_res = oversample.fit_resample(train_images.reshape(len(train_images), -1), y_train)
train_images_gray_res, y_train_gray_res = oversample.fit_resample(train_images_gray.reshape(len(train_images_gray), -1), y_train)


# In[5]:


from keras.preprocessing.image import ImageDataGenerator

# 1. Define the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=15,      # Random rotations between -15 to 15 degrees
    width_shift_range=0.1,  # Random shift in width by 10%
    height_shift_range=0.1, # Random shift in height by 10%
    shear_range=0.1,        # Shear transformations
    zoom_range=0.1,         # Random zooming up to 10%
    #horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill missing pixels using nearest neighbours
)

# 2. Apply Augmentation to Specific Classes
under_represented_classes = [20, 27, 40]
augmented_images = []
augmented_labels = []

for class_id in under_represented_classes:
    # Filter images of the specific class
    mask = y_train == class_id
    class_images = train_images[mask]
    
    # Augment the images. This will generate batches, so we loop through the generator
    # until we've produced a desired number of augmented samples for the class.
    # For instance, you could double the number of samples with num_augmented = len(class_images)
    num_augmented = 0
    for x_batch, y_batch in datagen.flow(class_images, np.zeros(len(class_images)), batch_size=32):
        augmented_images.extend(x_batch)
        augmented_labels.extend([class_id]*len(x_batch))
        num_augmented += len(x_batch)
        if num_augmented >= len(class_images):
            break

# Convert augmented data to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Add augmented data to the original data
train_images = np.concatenate([train_images, augmented_images], axis=0)
y_train = np.concatenate([y_train, augmented_labels], axis=0)

# Update the y_train_cat with the new y_train data
y_train_cat = to_categorical(y_train, num_classes=43)


# In[6]:


# After oversampling
class_counts_res = pd.Series(y_train_res).value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x=class_counts_res.index, y=class_counts_res.values)
plt.title("Class distribution after oversampling")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()


# In[9]:


# Reshape data
train_images_res = train_images_res.reshape(-1, 30, 30, 3)
train_images_gray_res = train_images_gray_res.reshape(-1, 30, 30, 1)
y_train_res_cat = to_categorical(y_train_res, num_classes=43)
y_train_gray_res_cat = to_categorical(y_train_gray_res, num_classes=43)


# In[16]:


from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Define and compile the color model
model_color = Sequential()
model_color.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 3)))
model_color.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_color.add(MaxPooling2D(pool_size=(2, 2)))
model_color.add(Dropout(0.25)) # Dropout layer added
model_color.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_color.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_color.add(MaxPooling2D(pool_size=(2, 2)))
model_color.add(Dropout(0.25)) # Dropout layer added
model_color.add(Flatten())
model_color.add(Dense(256, activation='relu'))
model_color.add(Dropout(0.5)) # Dropout layer added
model_color.add(Dense(43, activation='softmax'))
model_color.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the color model
history_color = model_color.fit(train_images_res, y_train_res_cat, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stopping])

# Define and compile the grayscale model
model_gray = Sequential()
model_gray.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
model_gray.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_gray.add(MaxPooling2D(pool_size=(2, 2)))
model_gray.add(Dropout(0.25)) # Dropout layer added
model_gray.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_gray.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_gray.add(MaxPooling2D(pool_size=(2, 2)))
model_gray.add(Dropout(0.25)) # Dropout layer added
model_gray.add(Flatten())
model_gray.add(Dense(256, activation='relu'))
model_gray.add(Dropout(0.5)) # Dropout layer added
model_gray.add(Dense(43, activation='softmax'))
model_gray.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the grayscale model
history_gray = model_gray.fit(train_images_gray_res, y_train_gray_res_cat, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stopping])


# In[17]:


# Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_color.history['accuracy'], label='Training Accuracy')
plt.plot(history_color.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Color Model - Training and Validation Accuracy')


# In[18]:


plt.subplot(1, 2, 2)
plt.plot(history_gray.history['accuracy'], label='Training Accuracy')
plt.plot(history_gray.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Grayscale Model - Training and Validation Accuracy')


# In[19]:


# Evaluate the color model on test data
test_loss, test_accuracy = model_color.evaluate(test_images, y_test_cat)
print('Color Model Test Accuracy:', test_accuracy)

# Evaluate the grayscale model on test data
test_images_gray = test_images_gray.reshape(-1, 30, 30, 1)
test_loss, test_accuracy = model_gray.evaluate(test_images_gray, y_test_cat)
print('Grayscale Model Test Accuracy:', test_accuracy)


# In[20]:


# Get predictions for both models
y_pred_color = np.argmax(model_color.predict(test_images), axis=-1)
y_pred_gray = np.argmax(model_gray.predict(test_images_gray), axis=-1)

# Print the classification report for both models
print("Classification report for color model:")
print(classification_report(y_test, y_pred_color))

print("Classification report for grayscale model:")
print(classification_report(y_test, y_pred_gray))


# Looking at the classification reports, the color model performed with slightly better macro precision and otherwise performed the same (for overall metrics). Increasing the patience would further separate the color model from the grayscale model (meaning I tried this and the model performed better), but the project requires we set the patience = 2 for early stopping. 
# 
# Additionally, while both models poorly identified pedestrians with excessive false negatives, the color model performed much better at identifying "Dangerous Curve Right" signs and roundabout mandatory signs.
# 
# Relatively strong F1 scores (> 0.9, as specified in the instructions) indicate a relatively healthy balance between precision and recall.
