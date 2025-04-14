import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader, CacheDataset
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
from monai.losses import DiceFocalLoss
from monai.losses import Dice
from monai.losses import FocalLoss
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

original_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# mlflow.set_experiment("CT-monai-flow")

data_dir = "../../data/DatasetChallenge"

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

# Validate the data before processing
print("Validating segmentation data...")
issues_found = validate_segmentation_data(data, max_class_expected=2)
if issues_found:
    print("⚠️ WARNING: Some segmentation labels exceed the expected range. Remapping will be applied.")

# Split into training and testing sets (80% train, 20% test)
if len(data) > 0:
    train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, random_state=42)
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
else:
    print("No data available for training/testing. Check your data directories.")
    train_data = []
    test_data = []


# Define MONAI transformations for training
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # Load NIfTI files
    EnsureChannelFirstd(keys=["image", "label"]),  # Add channel dimension
    ScaleIntensityd(keys=["image"]),  # Scale image intensities to [0, 1]
    ToTensord(keys=["image", "label"]),  # Convert to PyTorch tensors
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # RemapLabelsD(keys=["label"], max_classes=3),  # Apply same remapping to validation data
    ScaleIntensityd(keys=["image"]),
    # CropForegroundd(keys=["image", "label"], source_key="image"),
    ToTensord(keys=["image", "label"]),
])

# Create MONAI datasets and dataloaders
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

val_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SwinUNETR(
    img_size=(512, 512, 32), # input image size
    spatial_dims=3, # 3 dimensions  
    in_channels=1,
    out_channels=3, # background, tumor and metastasis
    use_checkpoint=True,
    feature_size = 64,
    drop_rate = 0.01,
    attn_drop_rate = 0.01
).to(device)

# Defining Loss & Optimizer
loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters, lr=0.0001)

# Define metric
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
num_epochs = 5
val_interval = 5  # Validate every 5 epochs
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in tqdm(train_loader):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        # Check for out-of-bounds values and fix them if necessary
        if torch.max(labels) >= 3:
            print(f"Warning: Found label value {torch.max(labels).item()} in batch. Clamping to 2.")
            labels = torch.clamp(labels, 0, 2)
        
        optimizer.zero_grad()
        # Get the predictions of the model
        outputs = model(inputs)
        # Compute the loss of the prediction
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                # Check and fix validation labels
                if torch.max(val_labels) >= 3:
                    val_labels = torch.clamp(val_labels, 0, 2)

                outputs = model(inputs)

                # Convert outputs to discrete classes
                val_outputs = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute metric
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved new best model")
            print(f"Current epoch: {epoch + 1}, Average Dice: {metric:.4f}")
            print(f"Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
