#!/usr/bin/env python3

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import copy
import sys
from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from mytraining import create_model


def setup(rank, world_size, port='12355'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank} initialized on GPU {rank}")

def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_worker(rank, world_size, model_params, data_name, model_name, preprocessed_name, val_fold):
    """Main distributed training function that runs on each GPU."""
    try:
        print(f"Starting worker on rank {rank}/{world_size}")
        
        # Setup distributed environment
        setup(rank, world_size)
        
        # Import the model here to avoid issues with multiprocessing
        sys.path.append('/home/samy/epita/pinkcc/PinkCC-PinkPanthers/models/ovseg2/src')
        
        # Deep copy model parameters to avoid sharing between processes
        model_params = copy.deepcopy(model_params)
        
        # Modify parameters for distributed training
        modify_params_for_distributed(model_params, rank, world_size)
        
        print(f"Rank {rank}: Creating model...")
        
        # Create model
        model = create_model(
            val_fold=val_fold,
            data_name=data_name,
            model_name=model_name,
            preprocessed_name=preprocessed_name,
            model_params=model_params,
            distributed=True  # Indicate that this is a distributed setup
        )
        
        print(f"Rank {rank}: Moving model to GPU...")
        
        # Move model to correct GPU
        device = torch.device(f'cuda:{rank}')
        model.network = model.network.to(device)
        
        print(f"Rank {rank}: Wrapping with DDP...")
        
        # Wrap with DistributedDataParallel
        ddp_network = DDP(
            model.network,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False  # Set to True if you have unused parameters
        )
        model.network = ddp_network
        # CRITICAL: Update the training object's network reference
        if hasattr(model, 'training') and hasattr(model.training, 'network'):
            model.training.network = ddp_network
        
        # Also update any other objects that might hold network references
        if hasattr(model, 'prediction') and hasattr(model.prediction, 'network'):
            model.prediction.network = ddp_network
            
        if hasattr(model, 'postprocessing') and hasattr(model.postprocessing, 'network'):
            model.postprocessing.network = ddp_network

        # Ensure training object knows about distributed setup
        if hasattr(model.training, '__dict__'):
            model.training.distributed = True
            model.training.rank = rank
            model.training.world_size = world_size
        
        print(f"Rank {rank}: Starting training...")
        
        # Start training
        model.training.train()
        
        # Only rank 0 does validation to avoid conflicts
        if rank == 0 and val_fold < model_params['data']['n_folds']:
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
    
    # Scale learning rate with world size (linear scaling rule)
    if 'training' in model_params and 'opt_params' in model_params['training']:
        if 'lr' in model_params['training']['opt_params']:
            original_lr = model_params['training']['opt_params']['lr']
            scaled_lr = original_lr * world_size
            model_params['training']['opt_params']['lr'] = scaled_lr
            print(f"Rank {rank}: Scaled learning rate from {original_lr} to {scaled_lr}")
    
    # Adjust batch size per GPU (keep total effective batch size the same)
    original_batch_size = model_params['data']['trn_dl_params']['batch_size']
    per_gpu_batch_size = max(1, original_batch_size // world_size)
    model_params['data']['trn_dl_params']['batch_size'] = per_gpu_batch_size
    
    val_batch_size = model_params['data']['val_dl_params']['batch_size']
    per_gpu_val_batch_size = max(1, val_batch_size // world_size)
    model_params['data']['val_dl_params']['batch_size'] = per_gpu_val_batch_size
    
    print(f"Rank {rank}: Train batch size per GPU: {per_gpu_batch_size} (total: {per_gpu_batch_size * world_size})")
    print(f"Rank {rank}: Val batch size per GPU: {per_gpu_val_batch_size} (total: {per_gpu_val_batch_size * world_size})")
    
    # Enable distributed flags
    model_params['data']['trn_dl_params']['distributed'] = True
    model_params['data']['val_dl_params']['distributed'] = True
    model_params['training']['distributed'] = True
    
    return model_params

def main():
    parser = argparse.ArgumentParser(description='Distributed Training for OvSeg')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--data_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='pagnoux_distributed')
    parser.add_argument('--preprocessed_name', type=str, default='preprocessed')
    parser.add_argument('--val_fold', type=int, default=1)
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run distributed training.")
        return
    
    # Determine world size
    available_gpus = torch.cuda.device_count()
    world_size = args.world_size if args.world_size is not None else available_gpus
    world_size = min(world_size, available_gpus)
    
    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) available. Distributed training needs at least 2 GPUs.")
        print("Running single GPU training instead...")
        world_size = 1
    
    print(f"Starting distributed training:")
    print(f"  - World size: {world_size}")
    print(f"  - Available GPUs: {available_gpus}")
    for i in range(world_size):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Import model parameters
    sys.path.append('/home/samy/epita/pinkcc/PinkCC-PinkPanthers/models/ovseg2')
    from mytraining import model_params
    
    if world_size == 1:
        # Single GPU fallback
        print("Running single GPU training...")
        train_worker(0, 1, model_params, args.data_name, args.model_name, args.preprocessed_name, args.val_fold)
    else:
        # Multi-GPU distributed training
        print("Spawning distributed training processes...")
        mp.spawn(
            train_worker,
            args=(world_size, model_params, args.data_name, args.model_name, args.preprocessed_name, args.val_fold),
            nprocs=world_size,
            join=True
        )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
