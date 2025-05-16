import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
import os
import json
# import matplotlib.pyplot as plt

def compute_f1_metric(y_true, y_pred): # Giữ nguyên hàm này
    y_true_labels = tf.argmax(y_true, axis=1)
    y_pred_labels = tf.argmax(y_pred, axis=1)
    def sklearn_f1_score(true_labels, pred_labels):
        return f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    return tf.numpy_function(sklearn_f1_score, [y_true_labels, y_pred_labels], tf.double)

def train_resnet_h5(): # Đổi tên hàm
    # --- Ưu tiên sử dụng GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Đang huấn luyện trên GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("Không tìm thấy GPU, đang sử dụng CPU.")

    # --- Thông số cấu hình ---
    data_dir = 'D:/UEH/ml/y/data/classification_dataset_augmented'
    img_width, img_height = 224, 224
    batch_size = 16
    num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
    print(f"Số lớp phát hiện được: {num_classes}")

    # --- 1. Chuẩn bị dữ liệu ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        zoom_range=[0.7, 1.3],
        shear_range=0.3,
        channel_shift_range=30.0,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'), target_size=(img_width, img_height),
        batch_size=batch_size, class_mode='categorical', shuffle=True)
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'), target_size=(img_width, img_height),
        batch_size=batch_size, class_mode='categorical', shuffle=False)

    # --- Class weights ---
    labels = train_generator.classes
    class_weights_computed = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights_computed))
    print(f"Class weights được sử dụng: {class_weight_dict}")

    # --- 2. Xây dựng mô hình ---
    base_model = ResNet50V2(weights='imagenet', include_top=False,
                            input_shape=(img_width, img_height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name="predictions")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # --- 3. Huấn luyện ban đầu ---
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), compute_f1_metric])
    model.summary()

    if not os.path.exists('models'):
        os.makedirs('models')
    checkpoint_initial_path = 'models/resnet50v2_initial.h5' # THAY ĐỔI ĐUÔI FILE
    checkpoint_initial = ModelCheckpoint(checkpoint_initial_path, monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1, save_weights_only=False)
    early_stopping_initial = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    lr_scheduler_initial = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    print("--- Bắt đầu huấn luyện các lớp mới (Initial Training - ResNet50V2) ---")
    history_initial = model.fit(
        train_generator, epochs=50, validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=[early_stopping_initial, lr_scheduler_initial, checkpoint_initial], verbose=1)

    history_initial_dict = {}
    for key, value in history_initial.history.items():
        history_initial_dict[key] = [float(v.numpy()) if hasattr(v, 'numpy') else float(v) for v in value]
    with open('training_history_resnet_initial.json', 'w') as f:
        json.dump(history_initial_dict, f, indent=4)
    print(f"Lịch sử huấn luyện ban đầu (ResNet50V2) đã được lưu vào training_history_resnet_initial.json")

    print(f"Tải lại mô hình tốt nhất từ: {checkpoint_initial_path}")
    model = tf.keras.models.load_model(checkpoint_initial_path, custom_objects={'compute_f1_metric': compute_f1_metric})

    # --- 4. Tinh chỉnh mô hình ---
    print("--- Bắt đầu tinh chỉnh mô hình (Fine-tuning - ResNet50V2) ---")
    fine_tune_at_layer_count = -30
    for layer in base_model.layers[fine_tune_at_layer_count:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
             layer.trainable = True

    model.compile( # Biên dịch lại
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), compute_f1_metric])
    model.summary()

    checkpoint_finetune_path = 'models/resnet50v2_finetuned.h5' # THAY ĐỔI ĐUÔI FILE
    checkpoint_fine_tune = ModelCheckpoint(checkpoint_finetune_path, monitor='val_accuracy',
                                           save_best_only=True, mode='max', verbose=1, save_weights_only=False)
    early_stopping_fine_tune = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_scheduler_fine_tune = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)

    history_fine_tune = model.fit(
        train_generator, epochs=50, validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=[early_stopping_fine_tune, lr_scheduler_fine_tune, checkpoint_fine_tune],
        initial_epoch=0, verbose=1)

    history_fine_tune_dict = {}
    for key, value in history_fine_tune.history.items():
        history_fine_tune_dict[key] = [float(v.numpy()) if hasattr(v, 'numpy') else float(v) for v in value]
    with open('training_history_resnet_fine_tune.json', 'w') as f:
        json.dump(history_fine_tune_dict, f, indent=4)
    print(f"Lịch sử fine-tuning (ResNet50V2) đã được lưu vào training_history_resnet_fine_tune.json")

    print(f"Tải lại mô hình tốt nhất từ: {checkpoint_finetune_path}")
    model = tf.keras.models.load_model(checkpoint_finetune_path, custom_objects={'compute_f1_metric': compute_f1_metric})

    # --- 5. Đánh giá mô hình cuối cùng ---
    print("--- Đánh giá mô hình cuối cùng (ResNet50V2) trên tập validation ---")
    val_generator.reset()
    val_loss, val_acc, val_precision, val_recall, val_f1 = model.evaluate(val_generator, verbose=1)
    print(f"Final Validation Loss (ResNet50V2): {val_loss:.4f}")
    print(f"Final Validation Accuracy (ResNet50V2): {val_acc*100:.2f}%")
    # ... (các print khác) ...
    y_pred_probabilities = model.predict(val_generator)
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    y_true_labels = val_generator.classes
    final_f1_sklearn = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    print(f"Final Validation F1-Score (Macro - scikit-learn, ResNet50V2): {final_f1_sklearn:.4f}")

if __name__ == "__main__":
    train_resnet_h5()