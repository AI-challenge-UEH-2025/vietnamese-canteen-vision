import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from sklearn.metrics import f1_score

def compute_f1_metric(y_true, y_pred):
    """Compute macro F1-score as a metric for Keras."""
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return f1_score(y_true.numpy(), y_pred.numpy(), average='macro')

class FoodClassifier:
    def __init__(self, model_path='models/resnet50v2_finetuned.h5', class_names=None):
        self.model = load_model(model_path, custom_objects={'compute_f1_metric': compute_f1_metric})
        self.class_names = class_names
        self.img_size = (224, 224)

    def classify(self, image_path):
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            with tf.device('/GPU:0'):
                predictions = self.model.predict(img_array)
            predicted_class = self.class_names[np.argmax(predictions[0])]
            return predicted_class
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return None