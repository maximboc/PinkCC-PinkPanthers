import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
# import mlflow

original_dir = os.getcwd()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import MapTransform  # Add this import

from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.ndimage import zoom
from tqdm import tqdm

# Set deterministic training for reproducibility
set_determinism(seed=0)
# mlflow.set_experiment("CT-monai-flow")

data_dir = "../../data/DatasetChallenge"
print(os.getcwd())
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


# Custom transform to remap label values
class RemapLabelsD(MapTransform):
    def __init__(self, keys, max_classes=3):
        super().__init__(keys)
        self.max_classes = max_classes
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Get the label data
            label_data = d[key]
            
            # Ensure values are within expected range
            unique_vals = torch.unique(label_data) if torch.is_tensor(label_data) else np.unique(label_data)
            
            # Check if remapping is needed
            if len(unique_vals) > self.max_classes or (torch.is_tensor(label_data) and torch.max(label_data) >= self.max_classes) or (not torch.is_tensor(label_data) and np.max(label_data) >= self.max_classes):
                # Create a mapping for unique values
                sorted_vals = sorted(unique_vals.tolist() if torch.is_tensor(unique_vals) else unique_vals)
                mapping = {val: min(i, self.max_classes-1) for i, val in enumerate(sorted_vals)}
                
                # Apply mapping
                if torch.is_tensor(label_data):
                    # For PyTorch tensors
                    result = torch.zeros_like(label_data)
                    for orig_val, new_val in mapping.items():
                        result[label_data == orig_val] = new_val
                else:
                    # For numpy arrays
                    result = np.zeros_like(label_data)
                    for orig_val, new_val in mapping.items():
                        result[label_data == orig_val] = new_val
                
                d[key] = result
                
        return d


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
mskcc_data = get_paired_data(ct_mskcc_dir, seg_mskcc_dir)
tcga_data = get_paired_data(ct_tcga_dir, seg_tcga_dir)
data = mskcc_data + tcga_data
print(len(data))

# Validate the data before processing
print("Validating segmentation data...")
issues_found = validate_segmentation_data(data, max_class_expected=2)
if issues_found:
    print("⚠️ WARNING: Some segmentation labels exceed the expected range. Remapping will be applied.")

# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# Define MONAI transformations for training
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # Load NIfTI files
    EnsureChannelFirstd(keys=["image", "label"]),  # Add channel dimension
    RemapLabelsD(keys=["label"], max_classes=3),  # Remap labels to range [0,2]
    ScaleIntensityd(keys=["image"]),  # Scale image intensities to [0, 1]
    CropForegroundd(keys=["image", "label"], source_key="image"),  # Crop foreground based on non-zero values in image
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=[96, 96, 32],  # adjust based on your GPU memory and desired patch size
        pos=1,
        neg=1,
        num_samples=4,
        image_key="image",
    ),  # Random crop patches
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),  # Random rotation
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),  # Random flipping
    ToTensord(keys=["image", "label"]),  # Convert to PyTorch tensors
])

# Define simpler transformations for validation (without augmentations)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    RemapLabelsD(keys=["label"], max_classes=3),  # Apply same remapping to validation data
    ScaleIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    ToTensord(keys=["image", "label"]),
])

# Create MONAI datasets and dataloaders
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

val_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define UNet model for segmentation
# Adjust in_channels and out_channels based on your data
# Assuming 1 channel CT input and 3 classes for segmentation (background, class1, class2)
model = UNet(
    spatial_dims=3,  # 3D segmentation
    in_channels=1,  # Single channel input (CT)
    out_channels=3,  # Three classes
    channels=(16, 32, 64, 128, 256),  # Feature map dimensions
    strides=(2, 2, 2, 2),  # Stride for each layer
    num_res_units=2,  # Number of residual units
).to(device)

# Define loss function and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define metric
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
num_epochs = 6
val_interval = 5  # Validate every 5 epochs
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

#with mlflow.start_run():
    # Log hyperparameters
"""
    mlflow.log_params({
        "lr": 1e-4,
        "batch_size": 2,
        "num_epochs": num_epochs,
        "val_interval": val_interval,
        "loss_function": "DiceLoss",
        "optimizer": "Adam",
        "architecture": "3D UNet",
        "input_channels": 1,
        "output_classes": 3,
    })
"""
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
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        #mlflow.log_metric("train_loss", epoch_loss, step=epoch)

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
                
                # Use sliding window inference for large 3D volumes
                roi_size = (96, 96, 32)  # match with the training crop size
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                
                # Convert outputs to discrete classes
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                
                # Compute metric
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            # Aggregate metrics
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            # mlflow.log_metric("val_dice", metric, step=epoch)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved new best model")
            print(f"Current epoch: {epoch + 1}, Average Dice: {metric:.4f}")
            print(f"Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")



# Plot loss and Dice over epochs
plt.figure("Training Metrics", (12, 6))

plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Epoch")
plt.plot(x, y)  # Fixed: Added plot function call

plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Epoch")
plt.plot(x, y)
plt.savefig("training_metrics.png")
# mlflow.log_artifact("training_metrics.png")
print(f"Training completed. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

# Load best model for inference
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

def visualize_prediction_3d_optimized(image_path, label_path, model, device, save_path=None, 
                                     downsample_factor=0.5, roi_center=None, roi_size=None):
    """
    Creates optimized 3D visualizations with downsampling and ROI selection.
    
    Args:
        image_path: Path to the input image (.nii or .nii.gz file)
        label_path: Path to the ground truth label (.nii or .nii.gz file)
        model: Trained MONAI model
        device: Computation device (CPU/GPU)
        save_path: Optional path to save the visualization
        downsample_factor: Factor to downsample the volumes (0.5 = half resolution)
        roi_center: Center coordinates of region of interest (optional)
        roi_size: Size of region of interest (optional)
    """
    # Load image and label
    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    
    # Ensure label values are within range [0,2]
    if np.max(label) >= 3:
        print(f"Remapping label values from range [0,{np.max(label)}] to [0,2]")
        # Create mapping for unique values
        unique_vals = np.unique(label)
        mapping = {val: min(i, 2) for i, val in enumerate(sorted(unique_vals))}
        remapped_label = np.zeros_like(label)
        for orig_val, new_val in mapping.items():
            remapped_label[label == orig_val] = new_val
        label = remapped_label
    
    # Extract region of interest if specified
    if roi_center is not None and roi_size is not None:
        # Convert to integers
        center = np.array(roi_center, dtype=int)
        size = np.array(roi_size, dtype=int)
        
        # Calculate ROI boundaries
        half_size = size // 2
        start = np.maximum(center - half_size, 0)
        end = np.minimum(center + half_size, np.array(image.shape))
        
        # Extract ROI
        roi_slice = tuple(slice(s, e) for s, e in zip(start, end))
        image = image[roi_slice]
        label = label[roi_slice]
    
    # Generate prediction on original resolution
    image_normalized = np.expand_dims(image, axis=0)  # Add channel dimension
    image_normalized = (image_normalized - image_normalized.min()) / (image_normalized.max() - image_normalized.min())
    image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = sliding_window_inference(image_tensor, (96, 96, 32), 4, model)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    # Downsample for visualization
    if downsample_factor < 1.0:
        print(f"Downsampling volumes for visualization (factor: {downsample_factor})...")
        image_small = zoom(image, downsample_factor, order=1)
        label_small = zoom(label, downsample_factor, order=0)  # order=0 to preserve label values
        pred_small = zoom(pred, downsample_factor, order=0)
    else:
        image_small = image
        label_small = label
        pred_small = pred
    
    print(f"Creating 3D visualization (downsampled shape: {image_small.shape})...")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Original Image with 3D volume rendering
    ax1 = fig.add_subplot(131, projection='3d')
    threshold = np.percentile(image_small, 75)
    print(f"Computing marching cubes for original image...")
    verts, faces, _, _ = marching_cubes(image_small, threshold)
    mesh = Poly3DCollection(verts[faces], alpha=0.25)
    mesh.set_edgecolor('none')
    mesh.set_facecolor('gray')
    ax1.add_collection3d(mesh)
    ax1.set_title("Original Image")
    set_axes_equal(ax1)
    ax1.set_xlim(0, image_small.shape[0])
    ax1.set_ylim(0, image_small.shape[1])
    ax1.set_zlim(0, image_small.shape[2])
    ax1.set_axis_off()
    
    # Ground Truth segmentation
    ax2 = fig.add_subplot(132, projection='3d')
    unique_labels = np.unique(label_small)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    print(f"Computing marching cubes for ground truth segmentation...")
    for i, label_val in enumerate(unique_labels):
        verts, faces, _, _ = marching_cubes(label_small == label_val, 0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        mesh.set_edgecolor('none')
        mesh.set_facecolor(colors[i])
        ax2.add_collection3d(mesh)
    
    ax2.set_title("Ground Truth")
    set_axes_equal(ax2)
    ax2.set_xlim(0, label_small.shape[0])
    ax2.set_ylim(0, label_small.shape[1])
    ax2.set_zlim(0, label_small.shape[2])
    ax2.set_axis_off()
    
    # Prediction segmentation
    ax3 = fig.add_subplot(133, projection='3d')
    unique_preds = np.unique(pred_small)
    unique_preds = unique_preds[unique_preds > 0]  # Skip background
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_preds)))
    print(f"Computing marching cubes for prediction segmentation...")
    for i, pred_val in enumerate(unique_preds):
        verts, faces, _, _ = marching_cubes(pred_small == pred_val, 0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        mesh.set_edgecolor('none')
        mesh.set_facecolor(colors[i])
        ax3.add_collection3d(mesh)
    
    ax3.set_title("Prediction")
    set_axes_equal(ax3)
    ax3.set_xlim(0, pred_small.shape[0])
    ax3.set_ylim(0, pred_small.shape[1])
    ax3.set_zlim(0, pred_small.shape[2])
    ax3.set_axis_off()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to {save_path}")
        # mlflow.log_artifact(save_path)
    
    plt.show()
    plt.close()

# Helper function to set equal aspect ratio for 3D plots
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Example usage
if len(test_data) > 0:
    test_case = test_data[0]
    # Use a smaller downsampling factor (0.25 = 1/4 resolution) for faster rendering
    # You can also specify an ROI to focus on a specific area
    # Example: roi_center=[128, 128, 80], roi_size=[100, 100, 60]
    visualize_prediction_3d_optimized(
        test_case["image"], 
        test_case["label"], 
        model, 
        device,
        save_path="3d_prediction_visualization.png",
        downsample_factor=0.25,  # Reduce to 1/4 resolution for faster rendering
        # Optionally specify ROI:
        # roi_center=[128, 128, 80],  # Center coordinates of region of interest
        # roi_size=[100, 100, 60]     # Size of region of interest
    )
    image_path = test_case["image"]
    if isinstance(image_path, str):
        print(f"3D visualization completed for test case: {os.path.basename(image_path)}")
    else:
        print("3D visualization completed.")
else:
    print("No test data available for visualization")

os.chdir(original_dir)
