import os
import cv2
import numpy as np
import albumentations as A
from shutil import copy2
import random

def create_augmentation_pipeline():
    """Define the augmentation pipeline with specified transformations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),  # Rotation between -15° and +15°
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=0.5)  # Brightness ±15%
    ])

def augment_image(image, transform):
    """Apply augmentation to a single image."""
    augmented = transform(image=image)
    return augmented['image']

def augment_class(class_dir, output_class_dir, current_count, target_count):
    """Augment images in a class directory to reach the target count."""
    os.makedirs(output_class_dir, exist_ok=True)
    
    # Copy original images
    images = os.listdir(class_dir)
    for img_name in images:
        src_path = os.path.join(class_dir, img_name)
        dst_path = os.path.join(output_class_dir, img_name)
        copy2(src_path, dst_path)

    # Define augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Calculate how many new images to generate
    images_to_generate = target_count - current_count
    if images_to_generate <= 0:
        print(f"Class {os.path.basename(class_dir)} already has {current_count} images, no augmentation needed.")
        return

    print(f"Augmenting class {os.path.basename(class_dir)}: {current_count} -> {target_count} images")

    # Generate augmented images
    generated_count = 0
    while generated_count < images_to_generate:
        # Randomly select an original image
        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Apply augmentation
        aug_image = augment_image(image, transform)
        
        # Save augmented image
        aug_img_name = f"aug_{generated_count}_{img_name}"
        aug_img_path = os.path.join(output_class_dir, aug_img_name)
        cv2.imwrite(aug_img_path, aug_image)
        generated_count += 1

def augment_dataset(input_dir, output_dir, target_min=200, target_max=220):
    """Augment the entire dataset to ensure each class has 100–120 images."""
    # Paths
    input_train_dir = os.path.join(input_dir, 'train')
    output_train_dir = os.path.join(output_dir, 'train')
    input_val_dir = os.path.join(input_dir, 'val')
    output_val_dir = os.path.join(output_dir, 'val')
    input_test_dir = os.path.join(input_dir, 'test')
    output_test_dir = os.path.join(output_dir, 'test')

    # Copy val and test directories unchanged
    if os.path.exists(input_val_dir):
        os.makedirs(output_val_dir, exist_ok=True)
        for class_name in os.listdir(input_val_dir):
            src_class_dir = os.path.join(input_val_dir, class_name)
            dst_class_dir = os.path.join(output_val_dir, class_name)
            os.makedirs(dst_class_dir, exist_ok=True)
            for img_name in os.listdir(src_class_dir):
                copy2(os.path.join(src_class_dir, img_name), os.path.join(dst_class_dir, img_name))
    
    if os.path.exists(input_test_dir):
        os.makedirs(output_test_dir, exist_ok=True)
        for class_name in os.listdir(input_test_dir):
            src_class_dir = os.path.join(input_test_dir, class_name)
            dst_class_dir = os.path.join(output_test_dir, class_name)
            os.makedirs(dst_class_dir, exist_ok=True)
            for img_name in os.listdir(src_class_dir):
                copy2(os.path.join(src_class_dir, img_name), os.path.join(dst_class_dir, img_name))

    # Augment train directory
    os.makedirs(output_train_dir, exist_ok=True)
    for class_name in os.listdir(input_train_dir):
        class_dir = os.path.join(input_train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Count images in the class
        current_count = len(os.listdir(class_dir))
        if current_count >= target_min:
            target_count = current_count  # No augmentation needed
        else:
            # Randomly select target between 100 and 120
            target_count = random.randint(target_min, target_max)

        output_class_dir = os.path.join(output_train_dir, class_name)
        augment_class(class_dir, output_class_dir, current_count, target_count)

def verify_dataset(output_dir):
    """Verify the number of images per class after augmentation."""
    train_dir = os.path.join(output_dir, 'train')
    print("\nVerification of augmented dataset:")
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"Class {class_name}: {num_images} images")

if __name__ == "__main__":
    input_dir = 'data/classification_dataset'
    output_dir = 'data/classification_dataset_augmented'
    
    # Augment the dataset
    augment_dataset(input_dir, output_dir, target_min=200, target_max=220)
    
    # Verify the result
    verify_dataset(output_dir)