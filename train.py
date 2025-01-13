import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Paths to your train and validation datasets
train_dir = 'path/to/train/dataset'
val_dir = 'path/to/validation/dataset'

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Create the model
model = create_model(num_classes=train_generator.num_classes)

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('plant_disease_model.h5')
