import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    ToTensord,
    AsDiscreted,
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import MapTransform
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.ndimage import zoom
from tqdm import tqdm
# import mlflow
set_determinism(seed=0)

# mlflow.set_experiment("CT-monai-flow")

data_dir = "PinkCC-PinkPanthers/data/DatasetChallenge"

# MSKCC
ct_mskcc_dir = os.path.join(data_dir, "CT", "MSKCC")
seg_mskcc_dir = os.path.join(data_dir, "Segmentation", "MSKCC")
print(ct_mskcc_dir)
# TCGA
ct_tcga_dir = os.path.join(data_dir, "CT", "TCGA")
seg_tcga_dir = os.path.join(data_dir, "Segmentation", "TCGA")


def get_paired_data(ct_dir, seg_dir):
    if not os.path.exists(ct_dir):
        print(f"CT directory not found: {ct_dir}")
        return []
    
    if not os.path.exists(seg_dir):
        print(f"Segmentation directory not found: {seg_dir}")
        return []
    
    ct_scans = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
    print(f"Found {len(ct_scans)} CT files in {ct_dir}")
    
    paired_data = []
    
    seg_filenames = os.listdir(seg_dir)
    
    for ct_scan in ct_scans:
        filename = os.path.basename(ct_scan)
        # Remove both .nii and .gz
        base_filename = os.path.splitext(os.path.splitext(filename)[0])[0]  

        matching_seg = None
        for seg_file in seg_filenames:
            if seg_file.startswith(base_filename) and seg_file.endswith('.nii.gz'):
                matching_seg = os.path.join(seg_dir, seg_file)
                # breaks if we found corresponding seg file
                break
        
        if matching_seg:
            paired_data.append({
                "image": ct_scan,
                "label": matching_seg
            })
        else:
            print(f"Warning : No segmentation found for {ct_scan}")
    
    return paired_data

# Function to check and validate label values
def validate_segmentation_data(data_list, max_class_expected=2):
    """Validate that segmentation labels don't exceed expected classes"""
    issues_found = False
    for item in data_list:
        label_path = item["label"]
        label_data = nib.load(label_path).get_fdata()
        unique_vals = np.unique(label_data)
        if np.max(unique_vals) > max_class_expected:
            print(f"Warning: {label_path} contains values up to {np.max(unique_vals)}")
            print(f"Unique values: {unique_vals}")
            issues_found = True
    return issues_found


# Get paired data from both sources
#mskcc_data = get_paired_data(ct_mskcc_dir, seg_mskcc_dir)
tcga_data = get_paired_data(ct_tcga_dir, seg_tcga_dir)
data = tcga_data # + mskcc_data 
print(len(data))
data = data #[:40]
# shape of data
#print(f"Shape of data: {data[0].shape}")


# Validate the data before processing
print("Validating segmentation data...")
#issues_found = validate_segmentation_data(data, max_class_expected=2)
#if issues_found:
#    print("⚠️ WARNING: Some segmentation labels exceed the expected range. Remapping will be applied.")

# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2,train_size=0.8,random_state=42)


print(f"Training samples: {len(train_data)}")
print(f"")
print(f"Testing samples: {len(test_data)}")

from monai.data.utils import pad_list_data_collate

from monai.transforms import Resized
# Define MONAI transformations for training
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(512, 512, 96)),
    ScaleIntensityd(keys=["image"]),
    AsDiscreted(keys=['label'], to_onehot=3),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(512, 512, 96)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])
print("Creating Caches")
# Create MONAI datasets and dataloaders
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  collate_fn=pad_list_data_collate)

val_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

model = SwinUNETR(
    img_size=(512, 512, 96), # image size
    spatial_dims=3, # 3 dimensions  
    in_channels=1,
    out_channels=3, # background, tumor and metastasis
    use_checkpoint=True
).to(device)
model = torch.nn.DataParallel(model)


loss_function = FocalLoss()

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

# Setup metric for evaluation
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training parameters
num_epochs = 10
best_metric = -1
best_metric_epoch = -1
val_interval = 5  # Validate every 5 epochs

# Post-processing transforms for prediction and labels
post_pred = AsDiscreted(keys="pred", argmax=True)
post_label = AsDiscreted(keys="label", to_onehot=3)  # 3 classes

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}")
    for batch_data in progress_bar:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / step})
    
    scheduler.step()
    epoch_loss /= step
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")
    
    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_dice = 0
            val_steps = 0
            
            for val_data in tqdm(val_loader, desc="Validation"):
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # Sliding window inference for large 3D volumes
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_images, roi_size, sw_batch_size, model, overlap=0.5
                )
                
                # Convert predictions and labels
                val_outputs_list_processed = post_pred({"pred": val_outputs})["pred"]
                val_labels_list_processed = post_label({"label": val_labels})["label"]
                val_outputs_list = [val_outputs_list_processed]
                val_labels_list = [val_labels_list_processed]

                
                # Compute metric
                dice_metric(y_pred=val_outputs_list, y=val_labels_list)
                val_steps += 1
            
            # Aggregate the final metric
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved new best model with Dice score: {best_metric:.4f}")
            
            print(
                f"Current epoch: {epoch + 1}, Mean Dice: {metric:.4f}, "
                f"Best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

print(f"Training completed. Best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}")

# Update your visualize_results and predict_segmentation functions
def visualize_results(model, data_loader, device, num_examples=3):
    """Visualize segmentation results"""
    model.eval()
    
    examples_visualized = 0
    with torch.no_grad():
        for val_data in data_loader:
            if examples_visualized >= num_examples:
                break
                
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            # Execute inference with smaller roi_size
            roi_size = (64, 64, 64)  # Smaller ROI for inference
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model, overlap=0.5
            )
            val_outputs = torch.softmax(val_outputs, dim=1)
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            
            # Convert to numpy for visualization
            val_images = val_images.cpu().numpy()
            val_labels = val_labels.cpu().numpy()
            val_outputs = val_outputs.cpu().numpy()
            
            # Plot middle slices
            for i in range(val_images.shape[0]):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Get middle slices for each dimension
                z_mid = val_images[i, 0].shape[0] // 2
                
                # Plot axial view (z-axis)
                axes[0, 0].imshow(val_images[i, 0, z_mid], cmap="gray")
                axes[0, 0].set_title("Image - Axial")
                axes[0, 1].imshow(val_labels[i, 0, z_mid], cmap="viridis")
                axes[0, 1].set_title("True Label - Axial")
                axes[0, 2].imshow(val_outputs[i, 0, z_mid], cmap="viridis")
                axes[0, 2].set_title("Prediction - Axial")
                
                # Plot coronal view (y-axis)
                y_mid = val_images[i, 0].shape[1] // 2
                axes[1, 0].imshow(val_images[i, 0, :, y_mid, :].T, cmap="gray")
                axes[1, 0].set_title("Image - Coronal")
                axes[1, 1].imshow(val_labels[i, 0, :, y_mid, :].T, cmap="viridis")
                axes[1, 1].set_title("True Label - Coronal")
                axes[1, 2].imshow(val_outputs[i, 0, :, y_mid, :].T, cmap="viridis")
                axes[1, 2].set_title("Prediction - Coronal")
                
                plt.tight_layout()
                plt.savefig(f"segmentation_result_{examples_visualized}.png")
                # plt.show()
                
                examples_visualized += 1
                if examples_visualized >= num_examples:
                    break

# Function to predict on new data
def predict_segmentation(model, image_path, device):
    """Predict segmentation for a new image"""
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])
    
    data = transform({"image": image_path})
    image = data["image"].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        output = sliding_window_inference(
            image, roi_size, sw_batch_size, model, overlap=0.5
        )
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1, keepdim=True)
        
    return output.cpu().numpy()

print("Visualizing sample results...")
model.load_state_dict(torch.load("best_model.pth"))
visualize_results(model, val_loader, device)
# Visualize results after training (uncomment to use)
# print("Visualizing sample results...")
# model.load_state_dict(torch.load("best_model.pth"))
# visualize_results(model, val_loader, device)
