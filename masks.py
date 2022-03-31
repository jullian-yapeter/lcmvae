from torch import nn
import torch

## Only tested on single image yet ##

class PixelMask():
    def __init__(self, mask_ratio):
        self.mask_ratio = mask_ratio

    def __call__(self, imgs):
        """ 
        return (masked images, masks) 
        imgs: (N, C, H, W)
        """
        p = torch.rand_like(imgs[:, 0, :, :])
        masks = torch.where(p < self.mask_ratio, False, True)
        return imgs.masked_fill_(masks.unsqueeze(1), 0), masks



class PatchMask():
    """
    reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
    """
    def __init__(self, mask_ratio, patch_size):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, H*W*3, patch_size**2)
        """
        p = self.patch_size
        N, _, H, W = imgs.shape
        assert H == W and H % p == 0

        h = w = H // p
        imgs = imgs.reshape(N, 3, p, h, p, w)
        imgs = torch.einsum('ncphqw->nhwcpq', imgs)
        imgs = imgs.reshape(N, h * w * 3, p**2)
        return imgs

    def unpatchify(self, x):
        """
        x: (N, L=h*w*3, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        N, L, _ = x.shape
        p = self.patch_size
        h = w = int((L//3)**.5)
        assert h * w == L//3

        x = x.reshape(N, h, w, 3, p, p)
        x = torch.einsum('nhwcpq->ncphqw', x)
        imgs = x.reshape(N, 3, h * p, w * p)
        return imgs

    def random_masking(self, x):
        """
        x: [N, L, P], sequence
        """
        N, L, P = x.shape  # batch, length, patches
        p = torch.rand(N, P, device=x.device)
        mask = torch.where(p < self.mask_ratio, True, False)
        x_masked = x.masked_fill_(mask.unsqueeze(1), 0)
        return x_masked, mask

    def __call__(self, imgs):
        masked_imgs = self.patchify(imgs)
        masked_imgs, mask = self.random_masking(masked_imgs)
        return self.unpatchify(masked_imgs), mask



