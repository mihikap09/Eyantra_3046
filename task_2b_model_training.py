import tensorflow as tf
import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# Define your learning rate schedule function
def learning_rate_schedule(epoch):
    if epoch < 10:
        return 0.0001
    elif epoch < 20:
        return 0.00001
    else:
        return 0.000001
    
# Create the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(learning_rate_schedule)

# Load the pre-trained VGG-16 model without the top classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model by adding custom classification layers on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Adjust the number of units for your specific task

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for the validation set
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Specify the paths to your training and validation datasets
train_data_directory = "C://Users//HP//training"
validation_data_directory = "C://Users//HP//testing"

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load and augment validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model using the data generators
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // 32,
                    epochs=15,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // 32,
                    callbacks=[lr_scheduler]
                    )

# Save the model to a file
model.save("vgg16_model.keras")




