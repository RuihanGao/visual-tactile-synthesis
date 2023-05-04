import torch
import numpy as np

"""
This file contains commonly used loss functions for surface normal estimation.
Ref: https://github.com/MARSLab-UMN/TiltedImageSurfaceNormal/blob/master/losses.py
"""

# Change Log: 2022-09-28: set both the optimization loss and logging loss as the mean. I think it used sum for optimization because of the absoluate value if very small, but need to verify.
def compute_surface_normal_angle_error(real_normal, surface_normal_pred, mode='evaluate'):
    # TODO: can feed in the mask here
    sample_batched = {'Z': real_normal}
    if 'Z' in sample_batched:
        N, C, H, W = sample_batched['Z'].shape
        if 'mask' not in sample_batched:
            sample_batched['mask'] = torch.ones((N, H, W)).type(torch.LongTensor).to(sample_batched['Z'].device)
        mask = sample_batched['mask'] > 0
        mask = mask.detach()
        cosine_similarity = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
        # # Tried to sum over patches then mean across the batch. For future use.
        # print("In function compute_surface_normal_angle_error, cosine_similarity", cosine_similarity.shape) # [NxNT, H, W]
        # N, H, W = cosine_similarity.shape
        # cosine_similarity = cosine_similarity.view(-1, batch_size_G2, H, W) 
        # mask = mask.view(-1, batch_size_G2, H, W)
        # masked_similarity = cosine_similarity[mask]
        # raise ValueError("stop here")
        
        # print("check range of cosine similartiy")
        # print("cosine_similarity.min()", cosine_similarity.min(), "cosine_similarity.max()", cosine_similarity.max())

        if mode == 'evaluate':
            cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0)
            return torch.acos(cosine_similarity) * 180.0 / np.pi # convert to angle in degrees

        elif mode == 'train_L2_loss':
            return 1.0-torch.mean(cosine_similarity[mask]), 1.0-torch.mean(cosine_similarity[mask])

        elif mode == 'train_AL_loss':
            acos_mask = mask.float() \
                   * (cosine_similarity.detach() < 0.999).float() * (cosine_similarity.detach() > -0.999).float()
            acos_mask = acos_mask > 0.0
            return torch.mean(torch.acos(cosine_similarity[acos_mask])), torch.mean(torch.acos(cosine_similarity[acos_mask]))

        elif mode == 'train_TAL_loss':
            mask = sample_batched['mask'] > 0
            # Robust acos loss
            acos_mask = mask.float() \
                   * (cosine_similarity.detach() < 0.9999).float() * (cosine_similarity.detach() > 0.0).float()
            cos_mask = mask.float() * (cosine_similarity.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0
            optimize_loss = (torch.sum(torch.acos(cosine_similarity[acos_mask])) - torch.sum(cosine_similarity[cos_mask])) / (torch.sum(cos_mask) + torch.sum(acos_mask))
            logging_loss = optimize_loss.detach() 
            return optimize_loss, logging_loss
