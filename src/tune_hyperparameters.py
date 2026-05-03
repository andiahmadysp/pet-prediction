import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner as kt

# Konfigurasi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_size = (160, 160)
batch_size = 32
dataset_path = os.path.join(BASE_DIR, "data", "PetImages")
model_path = os.path.join(BASE_DIR, "models", "best_model.keras")
tuner_dir = os.path.join(BASE_DIR, "models", "tuner_dir")

# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    validation_split=0.2
)

# Hanya preprocessing untuk validasi
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)


def build_model(hp):
    """Build model dengan search space untuk hyperparameter tuning."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(160, 160, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    # Search: dropout rate layer pertama
    dropout_1 = hp.Float("dropout_1", min_value=0.3, max_value=0.7, step=0.2)
    x = Dropout(dropout_1)(x)

    # Search: jumlah unit dense layer
    dense_units = hp.Choice("dense_units", values=[128, 256, 512])
    x = Dense(dense_units, activation="relu")(x)

    # Search: apakah menggunakan batch normalization kedua
    if hp.Boolean("use_second_bn"):
        x = BatchNormalization()(x)

    # Search: dropout rate layer kedua
    dropout_2 = hp.Float("dropout_2", min_value=0.2, max_value=0.5, step=0.1)
    x = Dropout(dropout_2)(x)

    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Search: learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory=tuner_dir,
    project_name="pet_prediction_tuning"
)

stop_early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

print("=" * 60)
print("Fase 1: Hyperparameter Search dengan Hyperband")
print("=" * 60)

tuner.search(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[stop_early]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "=" * 60)
print("Best Hyperparameters ditemukan:")
print("=" * 60)
print(f"  Learning rate : {best_hps.get('learning_rate')}")
print(f"  Dropout 1     : {best_hps.get('dropout_1')}")
print(f"  Dropout 2     : {best_hps.get('dropout_2')}")
print(f"  Dense units   : {best_hps.get('dense_units')}")
print(f"  Second BN     : {best_hps.get('use_second_bn')}")
print("=" * 60)

# Build best model
model = tuner.hypermodel.build(best_hps)

# Callbacks untuk final training
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

print("\nFase 2: Melatih model terbaik (head only)...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# Fine-tuning
print("\nFase 3: Fine-tuning model terbaik...")
base_model = model.layers[1]  # MobileNetV2 layer
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

model.save(model_path)
print(f"\nModel tersimpan sebagai {model_path}")
