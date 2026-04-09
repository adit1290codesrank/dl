import numpy as np
import struct
import os
import random
from scipy.ndimage import shift, rotate, gaussian_filter, map_coordinates

def load_idx_images(filepath):
    print(f"Loading images from {filepath}...")
    with open(filepath,'rb') as f:
        magic,num_imgs,rows,cols=struct.unpack(">IIII",f.read(16))
        images=np.frombuffer(f.read(),dtype=np.uint8).reshape(num_imgs,rows,cols)
    return images

def load_idx_labels(filepath):
    print(f"Loading labels from {filepath}...")
    with open(filepath,'rb') as f:
        magic,num_labels=struct.unpack(">II",f.read(8))
        labels=np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_idx_images(filepath,images):
    print(f"Saving {len(images)} images to {filepath}...")
    with open(filepath,'wb') as f:
        f.write(struct.pack(">IIII",2051,len(images),28,28))
        f.write(images.tobytes())
        
def save_idx_labels(filepath,labels):
    print(f"Saving {len(labels)} labels to {filepath}...")
    with open(filepath,'wb') as f:
        f.write(struct.pack(">II",2049,len(labels)))
        f.write(labels.tobytes())

def elastic_deform(image, alpha=8.0, sigma=4.0):
    h, w = image.shape

    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h-1), np.clip(x + dx, 0, w-1)]

    warped = map_coordinates(image.astype(np.float32), coords, order=1, mode='constant', cval=0)
    return np.clip(warped, 0, 255).astype(np.uint8)

def apply_rotation(image, max_angle=10.0):

    angle = random.uniform(-max_angle, max_angle)
    rotated = rotate(image.astype(np.float32), angle, reshape=False, mode='constant', cval=0, order=1)
    return np.clip(rotated, 0, 255).astype(np.uint8)

def augment_dataset(images, labels, copies_per_image=2):
    print(f"Augmenting dataset... (Generating {copies_per_image} new copies per image)")
    
    new_images = []
    new_labels = []

    new_images.append(images)
    new_labels.append(labels)

    print(f"  Generating copy batch 1/{copies_per_image} — shifts...")
    shifted_batch = np.zeros_like(images)
    for i in range(len(images)):
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        shifted_batch[i] = shift(images[i], shift=(dy, dx), cval=0.0)
    new_images.append(shifted_batch)
    new_labels.append(labels)

    print(f"  Generating copy batch 2/{copies_per_image} — rotation + elastic warp...")
    elastic_batch = np.zeros_like(images)
    for i in range(len(images)):
        img = apply_rotation(images[i], max_angle=10.0)
        img = elastic_deform(img, alpha=8.0, sigma=4.0)
        elastic_batch[i] = img
    new_images.append(elastic_batch)
    new_labels.append(labels)

    final_images = np.concatenate(new_images, axis=0)
    final_labels = np.concatenate(new_labels, axis=0)
    
    print("Shuffling the expanded dataset...")
    indices = np.arange(len(final_images))
    np.random.shuffle(indices)
    
    return final_images[indices], final_labels[indices]

if __name__ == "__main__":
    TRAIN_IMG_IN  = "./data/emnist-balanced-train-images-idx3-ubyte"
    TRAIN_LBL_IN  = "./data/emnist-balanced-train-labels-idx1-ubyte"
    
    TRAIN_IMG_OUT = "./data/emnist-augmented-train-images.idx3"
    TRAIN_LBL_OUT = "./data/emnist-augmented-train-labels.idx1"

    print("=== Processing Train Set ===")
    train_imgs = load_idx_images(TRAIN_IMG_IN)
    train_lbls = load_idx_labels(TRAIN_LBL_IN)
    
    aug_train_imgs, aug_train_lbls = augment_dataset(train_imgs, train_lbls, copies_per_image=2)
    
    save_idx_images(TRAIN_IMG_OUT, aug_train_imgs)
    save_idx_labels(TRAIN_LBL_OUT, aug_train_lbls)

    print(f"\nDone! {len(aug_train_imgs)} total images saved.")