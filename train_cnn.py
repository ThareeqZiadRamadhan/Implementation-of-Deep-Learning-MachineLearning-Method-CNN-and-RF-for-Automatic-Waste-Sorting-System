import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
import os
import sys

# 1. Konfigurasi
dataset_dir = "archive"
batch_size = 32
img_size = (128, 128)
epochs = 15
fine_tune_epochs = 5
model_name = "models/model_cnn.keras"  # Simpan di folder models

# 2. Validasi dataset
print("=" * 60)
print("  TRAINING CNN - KLASIFIKASI SAMPAH")
print("=" * 60)

if not os.path.exists(dataset_dir):
    print(f"\n❌ Error: Folder dataset '{dataset_dir}' tidak ditemukan!")
    print("📁 Pastikan struktur folder seperti ini:")
    print("""
    archive/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
    """)
    sys.exit(1)

print(f"\n📂 Dataset ditemukan di: {dataset_dir}")

# Cek subfolder
try:
    subfolders = [f for f in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, f))]
    print(f"📁 Subfolder: {subfolders}")
    
    if len(subfolders) == 0:
        print("❌ Tidak ada subfolder kategori ditemukan!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error membaca folder: {e}")
    sys.exit(1)

# 3. Load dataset dengan error handling
print("\n📊 Loading dataset...")

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    print(f"✅ Kelas yang ditemukan: {class_names}")
    print(f"📊 Jumlah kelas: {num_classes}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("💡 Pastikan setiap subfolder berisi gambar (jpg/png)")
    sys.exit(1)

# Hitung jumlah data
train_size = tf.data.experimental.cardinality(train_ds).numpy()
val_size = tf.data.experimental.cardinality(val_ds).numpy()
print(f"📈 Data training: {train_size} batch (~{train_size * batch_size} gambar)")
print(f"📉 Data validasi: {val_size} batch (~{val_size * batch_size} gambar)")

# 4. Optimisasi pipeline data
print("\n🔧 Optimizing data pipeline...")
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# 6. Build Model dengan MobileNetV2
print("\n🏗️ Building model...")

IMG_SHAPE = img_size + (3,)

# Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

print(f"✅ Base model loaded: MobileNetV2")
print(f"📊 Base model layers: {len(base_model.layers)}")

# Build full model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = x = layers.Rescaling(scale=1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# 7. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# 8. Callbacks
os.makedirs(os.path.dirname(model_name), exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    ),
]

# 9. Training Phase 1: Feature Extraction
print("\n" + "=" * 60)
print("  PHASE 1: FEATURE EXTRACTION")
print("=" * 60)

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 1 completed!")
    print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    sys.exit(1)

# 10. Training Phase 2: Fine-tuning
print("\n" + "=" * 60)
print("  PHASE 2: FINE-TUNING")
print("=" * 60)

# Unfreeze top layers
base_model.trainable = True

fine_tune_at = 100
frozen_layers = 0
trainable_layers = 0

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    frozen_layers += 1

for layer in base_model.layers[fine_tune_at:]:
    trainable_layers += 1

print(f"🔒 Frozen layers: {frozen_layers}")
print(f"🔓 Trainable layers: {trainable_layers}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

total_epochs = len(history.epoch) + fine_tune_epochs

try:
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history.epoch),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 2 completed!")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Error during fine-tuning: {e}")
    history_fine = None

# 11. Save final model
print("\n" + "=" * 60)
print("  SAVING FINAL MODEL")
print("=" * 60)

try:
    # Kita akan paksa untuk menyimpan model lengkap.
    # Jika baris ini gagal, pesan error di bawah akan memberitahu kita alasannya.
    model.save(model_name)
    print(f"\n✅ Model berhasil disimpan dengan lengkap di: {model_name}")
    print("Sekarang Anda bisa melanjutkan ke langkah berikutnya (extract_features.py).")

except Exception as e:
    print(f"\n❌ GAGAL MENYIMPAN MODEL LENGKAP.")
    print("Ini adalah akar masalah dari semua error 'No model config' Anda sebelumnya.")
    print("Pesan error spesifik di bawah ini akan menjelaskan penyebabnya:")
    print("-" * 20)
    print(f"--> ERROR: {e}")
    print("-" * 20)
    print("\nSilakan perbaiki error di atas untuk bisa melanjutkan.")
# 12. Final Evaluation
print("\n" + "=" * 60)
print("  FINAL EVALUATION")
print("=" * 60)

try:
    final_loss, final_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"✅ Final validation accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"📊 Final validation loss: {final_loss:.4f}")
except Exception as e:
    print(f"⚠️ Could not evaluate model: {e}")

# 13. Plot training history
def plot_training_history(hist1, hist2=None):
    """Plot training history"""
    
    if hist2:
        acc = hist1.history['accuracy'] + hist2.history['accuracy']
        val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
        loss = hist1.history['loss'] + hist2.history['loss']
        val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
        fine_tune_start = len(hist1.history['accuracy'])
    else:
        acc = hist1.history['accuracy']
        val_acc = hist1.history['val_accuracy']
        loss = hist1.history['loss']
        val_loss = hist1.history['val_loss']
        fine_tune_start = None
    
    epochs_range = range(len(acc))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    
    if fine_tune_start:
        ax1.axvline(x=fine_tune_start, color='k', linestyle='--', 
                   alpha=0.7, label='Fine-tuning Start')
    
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    if fine_tune_start:
        ax2.axvline(x=fine_tune_start, color='k', linestyle='--', 
                   alpha=0.7, label='Fine-tuning Start')
    
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_cnn.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training plot saved: training_history_cnn.png")
    
    try:
        plt.show()
    except:
        pass  # Don't fail if display not available

print("\n📈 Creating training plots...")
plot_training_history(history, history_fine if 'history_fine' in locals() else None)

# Summary
print("\n" + "=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
print(f"✅ Model saved: {model_name}")
print(f"📊 Training plot: training_history_cnn.png")
print(f"🎯 Classes: {class_names}")
print(f"📈 Total epochs trained: {len(history.epoch) + (len(history_fine.epoch) if 'history_fine' in locals() and history_fine else 0)}")
print(f"🏆 Best validation accuracy: {max(history.history['val_accuracy'] + (history_fine.history['val_accuracy'] if 'history_fine' in locals() and history_fine else [])):.4f}")
print("\n🎉 CNN Training Completed Successfully!")
print("=" * 60)