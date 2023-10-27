from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Set the image size and number of classes
img_size = (150, 150)
num_classes = 5

# Create a ResNet-18 model with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Add custom layers for classification
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess your data (replace 'train_data_directory' with your data path)
train_dataset = train_datagen.flow_from_directory(
    'C://Users//HP//Downloads//training',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

# Load and preprocess your validation data (replace 'validation_data_directory' with your validation data path)
validation_dataset = valid_datagen.flow_from_directory(
    'C://Users//HP//Downloads//testing',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'
)

# Train the model with validation data
model.fit(
    train_dataset,
    epochs=10,  # You can adjust the number of epochs
    validation_data=validation_dataset,
    validation_steps=len(validation_dataset)
)

# Save the trained model if desired
model.save('resnet18_image_classification_with_validation.keras')
