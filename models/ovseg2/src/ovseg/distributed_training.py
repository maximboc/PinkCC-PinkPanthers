import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import copy

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model_params, data_name, model_name, preprocessed_name, val_fold):
    """Main distributed training function."""
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Import here to avoid issues with multiprocessing
    from ovseg.model.SegmentationModelV2 import SegmentationModelV2
    
    # Modify model parameters for distributed training
    model_params = modify_params_for_distributed(model_params, rank, world_size)
    
    try:
        # Create model
        model = SegmentationModelV2(
            val_fold=val_fold,
            data_name=data_name,  
            model_name=model_name,
            preprocessed_name=preprocessed_name,
            model_parameters=model_params
        )
        
        # Move model to GPU
        device = torch.device(f'cuda:{rank}')
        model.network = model.network.to(device)
        
        # Wrap model with DDP - this is crucial for distributed training
        model.network = DDP(
            model.network, 
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # Add this if you have unused parameters
        )
        
        # Setup distributed training parameters
        setup_distributed_training(model, rank, world_size)
        
        print(f"Rank {rank}: Starting training...")
        
        # Train
        model.training.train()
        
        # Evaluate only on rank 0
        if rank == 0 and hasattr(model_params['data'], 'n_folds') and val_fold < model_params['data']['n_folds']:
            print("Rank 0: Starting validation...")
            model.eval_validation_set()
    
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

def modify_params_for_distributed(model_params, rank, world_size):
    """Modify model parameters for distributed training."""
    model_params = copy.deepcopy(model_params)
    
    # Scale learning rate with world size (linear scaling rule)
    if 'training' in model_params and 'opt_params' in model_params['training']:
        if 'lr' in model_params['training']['opt_params']:
            original_lr = model_params['training']['opt_params']['lr']
            model_params['training']['opt_params']['lr'] = original_lr * world_size
            print(f"Scaled learning rate from {original_lr} to {model_params['training']['opt_params']['lr']}")
    
    # Keep total batch size the same across all GPUs
    # Each GPU will process batch_size/world_size samples
    original_batch_size = model_params['data']['trn_dl_params']['batch_size']
    per_gpu_batch_size = max(1, original_batch_size // world_size)
    model_params['data']['trn_dl_params']['batch_size'] = per_gpu_batch_size
    model_params['data']['val_dl_params']['batch_size'] = max(1, model_params['data']['val_dl_params']['batch_size'] // world_size)
    
    print(f"Rank {rank}: Batch size per GPU: {per_gpu_batch_size} (total effective batch size: {per_gpu_batch_size * world_size})")
    
    # Add distributed flags to data parameters
    model_params['data']['trn_dl_params']['distributed'] = True
    model_params['data']['val_dl_params']['distributed'] = True
    
    # Add distributed flag to training parameters
    if 'training' not in model_params:
        model_params['training'] = {}
    model_params['training']['distributed'] = True
    
    return model_params

def setup_distributed_training(model, rank, world_size):
    """Setup distributed-specific training configurations."""
    # Set distributed parameters directly on the training object
    if hasattr(model.training, '__dict__'):
        model.training.distributed = True
        model.training.rank = rank
        model.training.world_size = world_size
        print(f"Rank {rank}: Distributed training setup complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--data_name', type=str, default='YES')
    parser.add_argument('--model_name', type=str, default='pagnoux')
    parser.add_argument('--preprocessed_name', type=str, default='preprocessed')
    parser.add_argument('--val_fold', type=int, default=1)
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run distributed training.")
        exit(1)
    
    # Import model parameters without creating the model
    import sys
    sys.path.append('/home/samy/epita/pinkcc/PinkCC-PinkPanthers/models/ovseg2')
    from mytraining import model_params
    
    world_size = min(args.world_size, torch.cuda.device_count())
    print(f"Starting distributed training with {world_size} GPUs")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if world_size < 2:
        print("Warning: Distributed training with less than 2 GPUs. Consider using single GPU training.")
    
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    mp.spawn(
        train_distributed,
        args=(world_size, model_params, args.data_name, args.model_name, args.preprocessed_name, args.val_fold),
        nprocs=world_size,
        join=True
    )