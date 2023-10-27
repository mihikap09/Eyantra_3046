from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras.optimizers

# Define data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1/255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Augmentation: randomly rotate images up to 20 degrees
    width_shift_range=0.2,  # Augmentation: horizontally shift images
    height_shift_range=0.2,  # Augmentation: vertically shift images
    shear_range=0.2,  # Augmentation: shear transformation
    zoom_range=0.2,  # Augmentation: random zoom
    horizontal_flip=True,  # Augmentation: flip horizontally
    fill_mode='nearest'  # How to fill in newly created pixels
)

# Load and preprocess your data
train_dataset = train_datagen.flow_from_directory(
    'training/', target_size=(150, 150), batch_size=32, class_mode='categorical')

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))  # Add dropout to reduce overfitting
model.add(Dense(5, activation='softmax'))  # Output layer with 5 units for 5 classes

# Compile the model
optim = keras.optimizers.Adam(learning_rate=0.0015)
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=45)  # You can adjust the number of epochs

# Save the trained model if desired
model.save('my_trained_model.h5')
