from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions_combined import CE_dice_pyramid_loss, to_one_hot_encoding, \
    weighted_combined_pyramid_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch.distributed as dist


class SegmentationTraining(NetworkTraining):

    def __init__(self, *args,
                 prg_trn_sizes=None,
                 prg_trn_arch_params=None,
                 prg_trn_aug_params=None,
                 prg_trn_resize_on_the_fly=True,
                 n_im_channels:int = 1,
                 batches_have_masks=False,
                 mask_with_bin_pred=False,
                 stop_after_epochs=[],
                 early_stopping_patience=10,
                 early_stopping_min_delta=0.001,
                 early_stopping_metric='val_loss',
                 distributed=True,
                 **kwargs):
        # Distributed training parameters
        self.rank = 0 if not distributed else dist.get_rank()
        self.distributed = distributed
        self.world_size = 1 if not distributed else dist.get_world_size()
        
        super().__init__(*args, **kwargs)
        self.prg_trn_sizes = prg_trn_sizes
        self.prg_trn_arch_params = prg_trn_arch_params
        self.prg_trn_aug_params = prg_trn_aug_params
        self.prg_trn_resize_on_the_fly = prg_trn_resize_on_the_fly
        self.n_im_channels = n_im_channels
        self.batches_have_masks = batches_have_masks
        self.mask_with_bin_pred = mask_with_bin_pred
        self.stop_after_epochs = stop_after_epochs

         # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric

        # Early stopping state
        if self.early_stopping_patience is not None:
            self.best_metric_value = None
            self.patience_counter = 0
            self.early_stopped = False
        
        # now have fun with progressive training!
        self.do_prg_trn = self.prg_trn_sizes is not None
        if self.do_prg_trn:
            if not self.prg_trn_resize_on_the_fly:
                # if we're not resizing on the fly we have to store rescaled data
                self.prg_trn_store_rescaled_data()
            self.prg_trn_n_stages = len(self.prg_trn_sizes)
            assert self.prg_trn_n_stages > 1, "please use progressive training only if you have "\
                "more then one stage."
            self.prg_trn_epochs_per_stage = self.num_epochs // self.prg_trn_n_stages
            self.prg_trn_update_parameters()
        else:
            self.prg_trn_process_batch = nn.Identity()

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)
    """
    def compute_batch_loss(self, batch):

        batch = batch.cuda()
        batch = self.prg_trn_process_batch(batch)

        if self.augmentation is not None:
            with torch.no_grad():
                # in theory we shouldn't need this context, but I had weird memory leaks and
                # it doesn't hurt
                batch = self.augmentation(batch)

        # now let's get the arrays from the batch
        # the easiest one:
        yb = batch[:, -1:]
        if self.batches_have_masks:
            # when we have a mask in the batch tensor the channels are orderes as 
            # im_channel[s], mask_channel, label_channel ...
            xb = batch[:, :-2]
            mask = batch[:, -2:-1]
        else:
            # ... otherwise we have im_channel[s], label_channel
            xb = batch[:, :-1]
            mask = None

        if self.mask_with_bin_pred:
            # masking with the binary prediction just means multipying the previously acquired mask
            # with the last channel of the input tensor
            mask = mask * xb[: -1:]      

        
        # Handle DDP wrapper - access out_channels through module attribute
        if hasattr(self.network, 'module'):
            # This is a DDP wrapped model
            out_channels = self.network.module.out_channels
        else:
            # This is a regular model
            out_channels = self.network.out_channels

        # Convert target to one-hot encoding
        yb = to_one_hot_encoding(yb, out_channels)

        
        # Handle DDP wrapper
        network = self.network.module if hasattr(self.network, 'module') else self.network
        out = self.network(xb)
        loss = self.loss_fctn(out, yb, mask)

        return loss
    """
    def compute_batch_loss(self, batch):
        batch = batch.cuda()
        batch = self.prg_trn_process_batch(batch)

        # Extract input and target from batch
        xb = batch[:self.n_im_channels]
        yb = batch[self.n_im_channels:]

        # Forward pass
        pred = self.network(xb)

        # Get out_channels safely
        if hasattr(self, 'get_model_attr'):
            out_channels = self.get_model_attr('out_channels')
        elif hasattr(self.network, 'module'):
            out_channels = self.network.module.out_channels
        else:
            out_channels = self.network.out_channels

        # Convert target to one-hot encoding
        yb = to_one_hot_encoding(yb, out_channels)

        # Compute loss
        loss = self.loss_fctn(pred, yb)
        
        return loss
    def prg_trn_update_parameters(self):

        if self.epochs_done == self.num_epochs:
            return

        # compute which stage we are in atm
        self.prg_trn_stage = min([self.epochs_done // self.prg_trn_epochs_per_stage,
                                  self.prg_trn_n_stages - 1])

        # this is just getting the input patch size for the current stage
        # if we use grid augmentations the out shape of the augmentation is the input size
        # for the network. Else we can take the sampled size
        if self.prg_trn_aug_params is not None:
            if 'out_shape' in self.prg_trn_aug_params:
                print_shape = self.prg_trn_aug_params['out_shape'][self.prg_trn_stage]
            else:
                print_shape = self.prg_trn_sizes[self.prg_trn_stage]
        else:
            print_shape = self.prg_trn_sizes[self.prg_trn_stage]

        self.print_and_log('\nProgressive Training: '
                           'Stage {}, size {}'.format(self.prg_trn_stage, print_shape),
                           2)

        # now set the new patch size
        if self.prg_trn_resize_on_the_fly:

            if self.prg_trn_stage < self.prg_trn_n_stages - 1:
                # the most imporant part of progressive training: we update the resizing function
                # that should make the batches smaller
                self.prg_trn_process_batch = resize(self.prg_trn_sizes[self.prg_trn_stage],
                                                    self.network.is_2d)
            else:
                # here we assume that the last stage of the progressive training has the desired
                # size i.e. the size that the augmentation/the dataloader returns
                self.prg_trn_process_batch = nn.Identity()
        else:
            # we need to change the folder the dataloader loads from
            new_folders = self.prg_trn_new_folders_list[self.prg_trn_stage]
            self.trn_dl.dataset.change_folders_and_keys(new_folders, self.prg_trn_new_keys)
            if self.val_dl is not None:
                self.val_dl.dataset.change_folders_and_keys(new_folders, self.prg_trn_new_keys)
            self.prg_trn_process_batch = nn.Identity()

        # now alter the regularization
        if self.prg_trn_arch_params is not None:
            # here we update architectural paramters, this should be dropout and stochastic depth
            # rate
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            self.network.update_prg_trn(self.prg_trn_arch_params, h)

        if self.prg_trn_aug_params is not None:
            # here we update augmentation parameters. The idea is we augment more towards the
            # end of the training
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            self.print_and_log('changing augmentation paramters with h={:.4f}'.format(h))
            if self.augmentation is not None:
                self.augmentation.update_prg_trn(self.prg_trn_aug_params, h, self.prg_trn_stage)
            if self.trn_dl.dataset.augmentation is not None:
                self.trn_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h,
                                                                self.prg_trn_stage)
            if self.val_dl is not None:
                if self.val_dl.dataset.augmentation is not None:
                    self.val_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h,
                                                                    self.prg_trn_stage)

    def prg_trn_store_rescaled_data(self):

        # if we don't want to resize on the fly, e.g. because the CPU loading time is the
        # bottleneck, we will resize the full volumes and save them in .npy files
        # extensions for all folders
        str_fs = '_'.join([str(p) for p in self.prg_trn_sizes[-1]])
        extensions = []
        for ps in self.prg_trn_sizes[:-1]:
            extensions.append(str_fs + '->' + '_'.join([str(p) for p in ps]))
        # the scaling factors we will use for resizing
        scales = []
        for ps in self.prg_trn_sizes[:-1]:
            scales.append((np.array(ps) / np.array(self.prg_trn_sizes[-1])).tolist())

        # let's get all the dataloaders we have for this training
        dl_list = [self.trn_dl]
        if self.val_dl is not None:
            dl_list.append(self.val_dl)

        # first let's get all the folder pathes of where we want to store the resized volumes
        ds = dl_list[0].dataset

        # here we might create the folders
        prepp = ds.vol_ds.preprocessed_path
        # these are the folders we're considering for the training
        folders = [ds.vol_ds.folders[ds.vol_ds.keys.index(ds.image_key)],
                   ds.vol_ds.folders[ds.vol_ds.keys.index(ds.label_key)]]
        if self.batches_have_masks:
            folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.mask_key)])
        if hasattr(ds, 'prev_pred_key'):
            if ds.prev_pred_key is not None:
                folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.prev_pred_key)])
        # folders with all downsampled data
        all_fols = []
        for fol in folders:
            for ext in extensions:
                path_to_fol = os.path.join(prepp, fol+'_'+ext)
                all_fols.append(path_to_fol)
                if not os.path.exists(path_to_fol):
                    os.mkdir(path_to_fol)

        self.print_and_log('resize on the fly was disabled. Instead all resized volumes will be '
                           'saved at ' + prepp + ' in the following folders:')
        self.print_and_log(str(all_fols))
        self.print_and_log('Checking and converting now')
        # let's look at each dl
        for dl in dl_list:
            ds = dl.dataset
            # now we cycle through the dataset to see if there are scans we still need to resize
            for ind, scan in enumerate(ds.vol_ds.used_scans):
                convert_scan = np.any([not os.path.exists(os.path.join(fol, scan))
                                       for fol in all_fols])
                if convert_scan:
                    # at least one .npy file is missing, convert...
                    self.print_and_log('convert scan '+scan)
                    tpl = ds._get_volume_tuple(ind)
                    # let's convert only the image
                    im = tpl[0]
                    # name of the image folder, should be \'images\' most of the time
                    im_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.image_key)]
                    self._rescale_and_save_arr(im, scales, extensions, prepp,
                                               im_folder, scan, is_lb=False)

                    # now the label
                    lb = tpl[-1]
                    lb_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.label_key)]
                    self._rescale_and_save_arr(lb, scales, extensions, prepp,
                                               lb_folder, scan, is_lb=True)

                    if self.batches_have_masks:

                        mask = tpl[-2]
                        mask_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.mask_key)]
                        
                        self._rescale_and_save_arr(mask, scales, extensions, prepp,
                                                   mask_folder, scan, is_lb=True)
                        
                        # old code for cascade model
                        # here the predictions from the previous stages have been 
                        
                        if len(tpl) == 4:
                            # in this case we're in the second stage and also resize the
                            # prediction from the previous stage
                            prd = tpl[1]
                            prd_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.prev_pred_key)]
                            
                            self._rescale_and_save_arr(prd, scales, extensions, prepp,
                                                        prd_folder, scan, is_lb=True)
                    
                    else:
                        if len(tpl) == 3:
                            # in this case we're in the second stage and also resize the
                            # prediction from the previous stage
                            prd = tpl[1]
                            prd_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.prev_pred_key)]
                            
                            self._rescale_and_save_arr(prd, scales, extensions, prepp,
                                                        prd_folder, scan, is_lb=True)

        # now we need the new_keys and new_folders for each stage to update the datasets
        self.prg_trn_new_keys = [ds.image_key, ds.label_key]
        folders = [ds.vol_ds.folders[ds.vol_ds.keys.index(ds.image_key)],
                   ds.vol_ds.folders[ds.vol_ds.keys.index(ds.label_key)]]
        
        if hasattr(ds, 'prev_pred_key'):
            if ds.prev_pred_key is not None:
                self.prg_trn_new_keys.append(ds.prev_pred_key)
                folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.prev_pred_key)])
                
        # if ds.prev_pred_key is not None:
        #     self.prg_trn_new_keys.append(ds.prev_pred_key)
        #     folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.prev_pred_key)])
        if self.batches_have_masks:
            self.prg_trn_new_keys.append(ds.mask_key)
            folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.mask_key)])
            
        self.prg_trn_new_folders_list = []
        for ext in extensions:
            self.prg_trn_new_folders_list.append([fol+'_'+ext for fol in folders])
        self.prg_trn_new_folders_list.append(folders)
        self.print_and_log('Done!', 1)

    def _rescale_and_save_arr(self, im, scales, extensions, path, folder, scan, is_lb):
        dtype = im.dtype
        n_dims = len(im.shape)
        # add additional axes for the interpolation. Torch wants that!
        if n_dims == 3:
            im = im[np.newaxis, np.newaxis]
        elif n_dims == 4:
            im = im[np.newaxis]
        else:
            raise ValueError('Got loaded image that is not 3d or 4d?')

        # interpolation modu
        mode = 'nearest' if is_lb else 'trilinear'
        im = torch.from_numpy(im).type(torch.float).to(self.dev)
        for scale, ext in zip(scales, extensions):
            im_rsz = F.interpolate(im, scale_factor=scale, mode=mode)
            im_rsz = im_rsz.cpu().numpy().astype(dtype)
            if n_dims == 3:
                im_rsz = im_rsz[0, 0]
            else:
                im_rsz = im_rsz[0]
            np.save(os.path.join(path, folder+'_'+ext, scan), im_rsz)
    
    def print_and_log(self, message, level=1):
        """Only print and log on rank 0 to avoid duplicate messages."""
        if self.rank == 0:
            super().print_and_log(message, level)

    def save_checkpoint(self):
        """Only save checkpoint on rank 0."""
        if self.rank == 0:
            super().save_checkpoint()

    def on_epoch_end(self):
        if self.distributed:
            dist.barrier()
        super().on_epoch_end()

        # Early stopping check
        if self.early_stopping_patience is not None and not self.early_stopped:
            # Get the current metric value
            if self.early_stopping_metric == 'val_loss':
                current_metric = self.val_losses[-1] if hasattr(self, 'val_losses') else None
                # For loss, lower is better
                is_better = lambda current, best: current < (best - self.early_stopping_min_delta)
                self.print_and_log(f'Checking early stopping with metric: {self.early_stopping_metric}')
            else:
                current_metric = None
                self.print_and_log(f'Unknown early stopping metric: {self.early_stopping_metric}')

            if current_metric is not None and not np.isnan(current_metric):
                if self.best_metric_value is None:
                    self.best_metric_value = current_metric
                    self.patience_counter = 0
                elif is_better(current_metric, self.best_metric_value):
                    self.best_metric_value = current_metric
                    self.patience_counter = 0
                    self.print_and_log(f'Early stopping: new best {self.early_stopping_metric}: {current_metric:.6f}')
                else:
                    self.patience_counter += 1
                    self.print_and_log(f'Early stopping: patience {self.patience_counter}/{self.early_stopping_patience}')

                    if self.patience_counter >= self.early_stopping_patience:
                        self.early_stopped = True
                        self.stop_training = True
                        self.print_and_log(f'Early stopping triggered after {self.patience_counter} epochs without improvement')

        if self.do_prg_trn:
            # if we do progressive training we update the parameters....
            if self.epochs_done % self.prg_trn_epochs_per_stage == 0:
                # ... if this is the right epoch for this
                self.prg_trn_update_parameters()

        # check if we want to stop the training after this epoch
        if self.epochs_done in self.stop_after_epochs:
            self.stop_training = True


class resize(nn.Module):

    def __init__(self, size, is_2d, n_im_channels=1):
        super().__init__()

        if len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        elif len(size) == 3:
            self.size = (int(size[0]), int(size[1]), int(size[2]))
        else:
            raise ValueError('Expected size to be of len 2 or 3, got {}'.format(len(size)))
            
        self.is_2d = is_2d
        self.n_im_channels=n_im_channels
        self.mode = 'bilinear' if self.is_2d else 'trilinear'

    def forward(self, batch):
        # first split the batch by channels, the first ones should always be the image
        # the others are masks e.g. input predictions, ground truth labels or loss masks
        im, mask = batch[:, :self.n_im_channels], batch[:, self.n_im_channels:]
        im = F.interpolate(im, size=self.size, mode=self.mode)
        mask = F.interpolate(mask, size=self.size)
        batch = torch.cat([im, mask], 1)
        return batch

class SegmentationTrainingV2(SegmentationTraining):
    
    def initialise_loss(self):
        self.loss_fctn = weighted_combined_pyramid_loss(**self.loss_params)
