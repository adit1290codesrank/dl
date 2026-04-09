import numpy as np
import struct
import os
import random
from scipy.ndimage import shift

def load_cifar10_batch(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Each record: 1 label byte + 3072 image bytes (CHW)
    records = data.reshape(-1, 3073)
    labels = records[:, 0]
    images = records[:, 1:].reshape(-1, 3, 32, 32)  # CHW
    return images, labels

def save_cifar10_batch(filepath, images, labels):
    print(f"Saving {len(labels)} records to {filepath}...")
    records = np.zeros((len(labels), 3073), dtype=np.uint8)
    records[:, 0] = labels
    records[:, 1:] = images.reshape(len(labels), -1)
    with open(filepath, 'wb') as f:
        f.write(records.tobytes())

def random_crop(image, pad=4):
    """
    Pad image by `pad` pixels on each side with zeros, then
    randomly crop back to 32×32. Standard CIFAR augmentation.
    image: CHW uint8
    """
    c, h, w = image.shape
    padded = np.pad(image, ((0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
    
    top  = random.randint(0, 2 * pad)
    left = random.randint(0, 2 * pad)
    return padded[:, top:top+h, left:left+w]

def horizontal_flip(image):
    """
    Flip image horizontally (mirror left-right).
    image: CHW uint8
    """
    return image[:, :, ::-1].copy()

def augment_cifar10(images, labels):
    """
    Generates one augmented copy per image:
      - random crop (pad 4, crop to 32x32)
      - random horizontal flip (50% chance)
    Returns originals + augmented concatenated and shuffled.
    """
    print(f"  Augmenting {len(images)} images...")
    aug_images = np.zeros_like(images)
    
    for i in range(len(images)):
        img = random_crop(images[i], pad=4)
        if random.random() < 0.5:
            img = horizontal_flip(img)
        aug_images[i] = img

    combined_images = np.concatenate([images, aug_images], axis=0)
    combined_labels = np.concatenate([labels, labels], axis=0)

    print("  Shuffling...")
    indices = np.arange(len(combined_labels))
    np.random.shuffle(indices)
    return combined_images[indices], combined_labels[indices]

if __name__ == "__main__":
    INPUT_DIR  = "./data/cifar-10-batches-bin"
    OUTPUT_DIR = "./data/cifar-10-augmented"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    batch_files = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ]

    all_images = []
    all_labels = []

    # Load all 5 batches
    for fname in batch_files:
        imgs, lbls = load_cifar10_batch(os.path.join(INPUT_DIR, fname))
        all_images.append(imgs)
        all_labels.append(lbls)

    all_images = np.concatenate(all_images, axis=0)  # 50000, 3, 32, 32
    all_labels = np.concatenate(all_labels, axis=0)  # 50000

    print(f"\nLoaded {len(all_labels)} total training images.")
    print("Augmenting...")

    aug_images, aug_labels = augment_cifar10(all_images, all_labels)

    print(f"\nFinal dataset: {len(aug_labels)} images (2x original).")

    # Save as a single merged batch file
    out_path = os.path.join(OUTPUT_DIR, "data_batch_aug.bin")
    save_cifar10_batch(out_path, aug_images, aug_labels)

    # Copy test batch unchanged — never augment test data
    import shutil
    test_src = os.path.join(INPUT_DIR,  "test_batch.bin")
    test_dst = os.path.join(OUTPUT_DIR, "test_batch.bin")
    shutil.copy(test_src, test_dst)
    print(f"Test batch copied unchanged to {test_dst}")

    print("\nDone!")