import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dataset(path = "./data/DatasetChallenge", amount = 10):
    """
    Loads a dataset of CT-Scans from a given path
    """
    images = []
    labels = []
    
    for root, _, files in os.walk(path):
        for name in files:
            if amount == 0:
                break  # Stop when the required amount is reached

            if name.endswith(".nii.gz"):  # Process only NIfTI files
                image_path = os.path.join(root, name)
                print(f"Loading: {image_path}")

                # Load the image
                img = nib.load(image_path)
                data = img.get_fdata()

                # Normalize between 0 and 1 (avoid division by zero)
                min_val, max_val = np.min(data), np.max(data)
                if max_val > min_val:  
                    data = (data - min_val) / (max_val - min_val)
                else:
                    data = np.zeros_like(data)  # Handle edge case of constant image

                # Store in list
                images.append(data)

                # Generate a label (e.g., 0 for MSKCC, 1 for TCGA)
                label = 0 if "MSKCC" in image_path else 1
                labels.append(label)

                amount -= 1  # Decrement count only after successful processing
        
        if amount == 0:
            break  # Stop if we have enough images

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"Loaded {len(images)} images.")
    return images, labels

def load_specific_image(image_path):
    """
    Loads a specific CT-Scans from a given path
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    return data

def plot_image(data, slice_index, cmap="gray"):
    """
    Plots a given slice of a CT scan
    """
    plt.imshow(data[:, :, slice_index], cmap=cmap)
    plt.title(f"Slice {slice_index}")
    plt.axis("off")
    plt.show()

    
