import torch.nn.functional as F

class Pad():
    def __init__(self, Y, X, spatial_patch_size):
        self.Y = Y
        self.X = X
        self.spatial_patch_size = spatial_patch_size
    
    def pad(self, x):
        pad_lat    = (self.spatial_patch_size - self.Y % self.spatial_patch_size) % self.spatial_patch_size
        pad_lon    = (self.spatial_patch_size - self.X % self.spatial_patch_size) % self.spatial_patch_size
        self.pad_top    = pad_lat // 2
        self.pad_bottom = pad_lat - self.pad_top
        self.pad_left   = pad_lon // 2
        self.pad_right  = pad_lon - self.pad_left
        #x = F.pad(x, (pad_left, pad_right, self.pad_top, self.pad_bottom), mode='reflect')
        B, V, T, Y, X = x.shape
        x = x.reshape(B * V * T, 1, Y, X)
        x = F.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), mode='reflect')
        x = x.reshape(B, V, T, Y + self.pad_top + self.pad_bottom, X + self.pad_left + self.pad_right)
        _, _, _, Y_pad, X_pad = x.shape
        return x, Y_pad, X_pad

    def crop(self, output, concepts):
        concepts = concepts[..., self.pad_top:self.pad_top+self.Y, self.pad_left:self.pad_left+self.X]           # (B, n_concepts, lead, Y, X)
        output   = output[..., self.pad_top:self.pad_top+self.Y, self.pad_left:self.pad_left+self.X] 
        return output, concepts