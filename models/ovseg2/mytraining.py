from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_UNet,get_model_params_3d_res_encoder_U_Net ,get_model_params_3d_from_preprocessed_folder
from earlystopping import EarlyStopping

import torch
# name of your raw dataset
data_name = 'test'
# same name as in the preprocessing script
preprocessed_name = 'preprocessed'
# give each model a unique name. This way the code will be able to identify them
model_name = 'huit_deux__deux_pagnoux'
# which fold of the training is performed?
# Example 5-fold cross-vadliation: CV folds are 0,1,...,4.
#                                  For each val_fold > 4 no CV is applied and 
#                                  100% of the training data is used
val_fold = 1

# now get hyper-parameters
# patch size used during (last stage of) training and inference
# z axis first, then xy
patch_size = [32, 216, 216]
# for standard UNet the number of inplane convolutions
n_2d_convs = 3
# wheter to use progressive learning or not. I often found it to have no
# effect on the performance, but reduces training time by up to 40%
use_prg_trn = True
# number of different foreground classes you want to segment
#n_fg_classes = 1
# it is recommended to perform the training with mixed precision (fp16)
# instead of full precision (fp32)
use_fp32 = False
# shapes introduced to the network during progressive learning
# rule of thumb: reduce total number of voxels by a factor of 4,3,2 in the
# first three stages and train last stage as usual
# be careful that the patch size is still executable for your U-Net
# e.g. a U-Net that downsamples 4 times inplane should have a patch size
# where the inplane size is divisible by 2**4 
out_shape = [
    [20, 128, 128],
    [22, 152, 152],
    [30, 192, 192],
    [32, 216, 216]
]


"""
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32,
                                
                                        out_shape=out_shape)
"""
model_params = get_model_params_3d_from_preprocessed_folder(data_name=data_name,preprocessed_name=preprocessed_name)
model_params['architecture'] = "unetresencoder"
params = model_params['network']

# Supprimer les clés non supportées
for key in ['kernel_sizes', 'kernel_sizes_up', 'n_pyramid_scales']:
    params.pop(key, None)

# Mettre à jour les bonnes clés
params.update({
    'in_channels': 1,
    'out_channels': 3,
    'is_2d': False,
    'filters': 32,
    'filters_max': 320,
    'conv_params': None,
    'nonlin_params': None,
    'block': 'res',  # ou ResBlock si besoin
    'z_to_xy_ratio': 6.25,
    'stochdepth_rate': 0,
    'n_blocks_list': [1, 2, 6, 3]
})

model_params["data"] = {
    "n_folds": 10,
    "fixed_shuffle": True,
    "ds_params": {},
    "trn_dl_params": {
        "patch_size": [32, 216, 216],
        "batch_size": 8,
        "num_workers": 5,
        "pin_memory": True,
        "epoch_len": 250,
        "p_bias_sampling": 0,
        "min_biased_samples": 1,
        "padded_patch_size": [32, 432, 432],
        "store_coords_in_ram": True,
        "memmap": "r",
        "n_im_channels": 1,
        "store_data_in_ram": False,
        "return_fp16": True,
        "n_max_volumes": None,
    },
    "val_dl_params": {
        "patch_size": [32, 216, 216],
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": True,
        "epoch_len": 8,
        "p_bias_sampling": 0,
        "min_biased_samples": 1,
        "padded_patch_size": [32, 432, 432],
        "store_coords_in_ram": True,
        "memmap": "r",
        "n_im_channels": 1,
        "store_data_in_ram": True,
        "return_fp16": True,
        "n_max_volumes": 16,
    },
    "keys": ["image", "label"],
    "folders": ["images", "labels"]
}


model_params["prediction"] = {
    "patch_size": [32, 216, 216],
    "batch_size": 1,
    "overlap": 0.5,
    "fp32": False,
    "patch_weight_type": "gaussian",
    "sigma_gaussian_weight": 0.125,
    "mode": "simple",
}




# CHANGE YOUR HYPER-PARAMETERS HERE! For example

# change batch size to 4
#model_params['data']['val_dl_params']['batch_size'] = 4
# change momentum
#model_params['training']['opt_params']['momentum'] = 0.98
# change weight decay
#model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['prg_trn_sizes'] =  [[ 20 ,256, 256],[ 22, 304 ,304],[ 30 ,384, 384],[ 32 ,432, 432]]
#model_params['training']['prg_trn_resize_on_the_fly'] = False
model_params['training']['num_epochs'] = 150

model_params['training']['prg_trn_resize_on_the_fly'] = True  # Enable this
model_params['data']['val_dl_params']['batch_size'] = 16
model_params["data"]["trn_dl_params"].update({
    "num_workers": 2,      # Reduce workers to avoid CPU/disk bottleneck
    "store_data_in_ram": True,  # Store data in RAM if you have enough
    "epoch_len": 50,       # Start with shorter epochs for testing
})

model_params['training']['early_stopping'] = {
    'patience': 20,
    'min_delta': 0.001,
    'monitor': 'val_loss'
}

# creat model object.
# this object holds all objects that define a deep neural network model
#   - preprocessing
#   - augmentation
#   - training
#   - slinding window evaluation
#   - postprocessing
#   - data and data sampling
#   - functions to iterate over datasets
#   - I'm sure I forgot something

model_params['preprocessed_path'] = "/home/user-data_challenge-33/data/preprocessed/SAMPLE/test_preprocessing"

print(model_params)

model = SegmentationModelV2(val_fold=val_fold,
                            data_name=data_name,
                            model_name=model_name,
                            preprocessed_name=preprocessed_name,
                            model_parameters=model_params,
                            use_multi_gpu=True)

# After the model is fully initialized, enable multi-GPU
"""
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Enabling multi-GPU support for {torch.cuda.device_count()} GPUs")
    model.network = torch.nn.DataParallel(model.network)
    print(f"Model wrapped with DataParallel")
    
    # Verify the wrapper
    print(f"Network type: {type(model.network)}")
    print(f"Available GPUs: {list(range(torch.cuda.device_count()))}")
"""
# Monkey patch the training to enable multi-GPU after GPU move
original_train = model.training.train

early_stopping = EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True
)

def patched_train():
    # Call original train which moves network to GPU
    model.training.network = model.training.network.to(model.training.dev)

    # NOW enable multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Enabling multi-GPU with {torch.cuda.device_count()} GPUs")
        device_ids = list(range(torch.cuda.device_count()))
        model.training.network = torch.nn.DataParallel(model.training.network, device_ids=device_ids)
        model.network = model.training.network  # Keep reference in sync
        print(f"Network wrapped with DataParallel: {device_ids}")

    # Continue with training
    model.training.enable_autotune()
    super(type(model.training), model.training).train()


def train_with_early_stopping():
    model.training.network = model.training.network.to(model.training.dev)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Enabling multi-GPU with {torch.cuda.device_count()} GPUs")
        device_ids = list(range(torch.cuda.device_count()))
        model.training.network = torch.nn.DataParallel(model.training.network, device_ids=device_ids)
        model.network = model.training.network
        print(f"Network wrapped with DataParallel: {device_ids}")
    
    model.training.enable_autotune()
    
    # Custom training loop with early stopping
    for epoch in range(model_params['training']['num_epochs']):
        # Train one epoch
        model.training.network.train()
        train_loss = model.training.train_one_epoch()
        
        # Validate
        model.training.network.eval()
        val_loss = model.training.validate_one_epoch()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check early stopping
        early_stopping(val_loss, model.training.network)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
    
    return model

# Replace the train method
model.training.train = train_with_early_stopping
# execute the trainig, simple as that!
# It will check for previous checkpoints and load them
print(f"Training script")
model.training.train()

# if cross-validation is applied you can evaluate the validation scans like this
# as stated above, val_fold > n_folds means using 100% training data e.g. no validation data
if val_fold < model_params['data']['n_folds']:
    model.eval_validation_set()

# uncomment to evaluate raw (test) dataset with the model
# model.eval_raw_dataset('MY_TEST_DATA')


# uncomment to evaluate ensemble e.g. of cross-validation models
# ens = SegmentationEnsembleV2(val_fold=list(range(model_params['data']['n_folds'])),
#                              model_name=model_name,
#                              data_name=data_name,
#                              preprocessed_name=preprocessed_name)
# typically I train all folds on different GPUs in parallel, this let's you wait
# until all trainings are done
# ens.wait_until_all_folds_complete()
# evaluate ensemble on test data
# ens.eval_raw_dataset('MY_TEST_DATA')


