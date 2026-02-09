import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

base_dir = "/path/to/cat-litter-monitor/training_data"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Get all image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
print(f"Total images: {len(image_files)}")

# Shuffle and split (80/20)
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}")

# Move files to train
for img_file in train_files:
    # Move image
    src_img = os.path.join(images_dir, img_file)
    dst_img = os.path.join(base_dir, "train/images", img_file)
    shutil.move(src_img, dst_img)
    
    # Move corresponding label
    label_file = img_file.replace('.jpg', '.txt')
    src_label = os.path.join(labels_dir, label_file)
    dst_label = os.path.join(base_dir, "train/labels", label_file)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"Warning: No label for {img_file}")

# Move files to val
for img_file in val_files:
    # Move image
    src_img = os.path.join(images_dir, img_file)
    dst_img = os.path.join(base_dir, "val/images", img_file)
    shutil.move(src_img, dst_img)
    
    # Move corresponding label
    label_file = img_file.replace('.jpg', '.txt')
    src_label = os.path.join(labels_dir, label_file)
    dst_label = os.path.join(base_dir, "val/labels", label_file)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"Warning: No label for {img_file}")

print("Done! Verifying counts:")
print(f"Train images: {len(os.listdir(os.path.join(base_dir, 'train/images')))}")
print(f"Train labels: {len(os.listdir(os.path.join(base_dir, 'train/labels')))}")
print(f"Val images: {len(os.listdir(os.path.join(base_dir, 'val/images')))}")
print(f"Val labels: {len(os.listdir(os.path.join(base_dir, 'val/labels')))}")
