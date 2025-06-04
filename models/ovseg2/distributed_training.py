import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
from mytraining import create_model, model_params


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
    
    # Create model
    model = SegmentationModelV2(
        val_fold=val_fold,
        data_name=data_name,  
        model_name=model_name,
        preprocessed_name=preprocessed_name,
        model_parameters=model_params
    )
    
    # Move model to GPU
    model.network = model.network.to(rank)
    
    # Wrap model with DDP
    model.network = DDP(model.network, device_ids=[rank])
    
    # Modify training for distributed setup
    setup_distributed_training(model, rank, world_size)
    
    # Train
    model.training.train()
    
    # Evaluate only on rank 0
    if rank == 0 and val_fold < model_params['data']['n_folds']:
        model.eval_validation_set()
    
    cleanup()


def modify_params_for_distributed(model_params, rank, world_size):
    """Modify model parameters for distributed training."""
    # Scale learning rate with world size
    if 'lr' in model_params['training']['opt_params']:
        model_params['training']['opt_params']['lr'] *= world_size
    
    # Adjust batch size per GPU
    original_batch_size = model_params['data']['trn_dl_params']['batch_size']
    model_params['data']['trn_dl_params']['batch_size'] = original_batch_size // world_size
    model_params['data']['val_dl_params']['batch_size'] = model_params['data']['val_dl_params']['batch_size'] // world_size
    
    # Adjust epoch length to maintain same number of iterations
    model_params['data']['trn_dl_params']['epoch_len'] = model_params['data']['trn_dl_params']['epoch_len'] // world_size
    
    return model_params


def setup_distributed_training(model, rank, world_size):
    """Setup distributed-specific training configurations."""
    # Override the training class to handle distributed training
    original_compute_batch_loss = model.training.compute_batch_loss
    
    def distributed_compute_batch_loss(batch):
        # Ensure batch is on correct device
        if hasattr(batch, 'cuda'):
            batch = batch.cuda(rank)
        return original_compute_batch_loss(batch)
    
    model.training.compute_batch_loss = distributed_compute_batch_loss
    model.training.rank = rank
    model.training.world_size = world_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--data_name', type=str, default='YES')
    parser.add_argument('--model_name', type=str, default='pagnoux')
    parser.add_argument('--preprocessed_name', type=str, default='preprocessed')
    parser.add_argument('--val_fold', type=int, default=1)
    
    args = parser.parse_args()
    
    # Import model parameters
    from mytraining import model_params
    
    world_size = args.world_size
    print(f"Starting distributed training with {world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(world_size, model_params, args.data_name, args.model_name, args.preprocessed_name, args.val_fold),
        nprocs=world_size,
        join=True
    )
