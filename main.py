import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_CSV = 'train.csv'
TEST_DIR = 'images/test/test_images'
SUBMISSION_FILE = 'submission.csv'
EPOCHS = 20  # Adjust as necessary

# Choose either .h5 or .keras format
# MODEL_FILE = 'best_model.h5'  # For .h5 format
MODEL_FILE = 'best_model.keras'  # For .keras format


# Step 1: Data Preparation
def load_data():
    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)

    # Ensure the 'class' column is of type string
    train_df['class'] = train_df['class'].astype(str)

    # Check unique classes
    unique_classes = train_df['class'].unique()
    print(f"Unique classes found: {len(unique_classes)}")

    return train_df, len(unique_classes)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array


# Step 2: Data Augmentation
def create_data_generators(train_df):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='images/train_images/',
        x_col='path',
        y_col='class',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='images/train_images/',
        x_col='path',
        y_col='class',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator


# Step 3: Model Selection
def create_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)  # Added batch normalization
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 4: Training the Model
def train_model(model, train_generator, val_generator):
    # Callbacks for better training control
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Use save_format='h5' only if using .h5 format
    # model_checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min', save_format='h5')  # Uncomment for .h5

    # For .keras format
    model_checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss',
                                       mode='min')  # Uncomment for .keras

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Train the model with validation data
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history


# Optional: Fine-Tuning the Model
def fine_tune_model(model, train_generator, val_generator):
    # Unfreeze some layers for fine-tuning
    for layer in model.layers[-30:]:  # Unfreeze the last 30 layers for fine-tuning
        layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train again with fine-tuning
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS // 2,  # Shorter fine-tuning phase
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    return history


# Step 5: Generating Predictions
def generate_predictions(model):
    submission_data = []  # List to collect prediction data

    for filename in os.listdir(TEST_DIR):
        if filename.endswith('.jpg'):
            img_path = os.path.join(TEST_DIR, filename)
            img_array = preprocess_image(img_path)
            img_array = np.expand_dims(img_array, axis=0)

            # Generate predictions
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)  # Class prediction

            # Append the prediction to the list
            submission_data.append({'path': os.path.join(TEST_DIR, filename), 'class': predicted_class})

    # Create a DataFrame from the collected data
    submission_df = pd.DataFrame(submission_data)

    # Ensure the format matches the expected output
    submission_df.to_csv(SUBMISSION_FILE, index=False, columns=['path', 'class'])


if __name__ == "__main__":
    # Load the data
    train_df, num_classes = load_data()

    # Create data generators
    train_generator, val_generator = create_data_generators(train_df)

    # Build and compile the model
    model = create_model(num_classes)

    # Train the model with validation data and callbacks
    history = train_model(model, train_generator, val_generator)

    # Fine-tune the model if needed
    fine_tune = False  # Set this to True if you want to fine-tune
    if fine_tune:
        fine_tune_model(model, train_generator, val_generator)

    # Generate predictions on the test data
    generate_predictions(model)
