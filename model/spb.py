import torch
import torch.nn as nn

class SpectralBlock(nn.Module):

    def __init__(self, dim, h, w, mlp_ratio):
        super().__init__()
              
        self.layer_norm_1 = nn.LayerNorm([h, w])
        self.spectralfft = SpectralFFT(dim = dim, h=h, w=w)
        self.layer_norm_2 = nn.LayerNorm([h, w])
        self.drop = nn.Dropout(0.2)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim)

        
    def forward(self, x):
        # print("x",x.shape)
        x = x.permute(0, 1, 2, 3)
        output = self.drop(self.mlp(self.layer_norm_2(self.spectralfft(self.layer_norm_1(x))))).permute(0, 3, 1, 2)
        # print("output",output.shape)
        return output + x
       
class SpectralFFT(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        if dim == 512:
            self.h = int(h) # H
            self.w = int(w/2 + 1) # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 256:
            self.h = int(h) # H
            self.w = int(w / 2 + 1) # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 128:
            self.h = int(h) # H
            self.w = int(w / 2 + 1) # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 64:
            self.h = int(h) # H
            self.w = int(w / 2 + 1) # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
     
    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size
        # x = x.view(B, C, a, b)
        x = x.to(torch.float32) # x [2,128,128,128]
        # print('x before', x.shape)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho') # x [2,128,128,65]
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight', weight.shape)
        # print('x', x.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features=None,
                 drop=0.2):
        super().__init__()
    
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
    
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    
class CrackAttentionModule(nn.Module):
    
    """Position attention module
    
    """

    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.visual_pre = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): [bs, c, h, w]

        Returns:
            _type_: _description_
        """
        bs, c, h, w = x.shape
        
        visual_feats = self.visual_pre(x).view(bs, -1, h * w)
        
        attention = self.softmax(torch.bmm(visual_feats.permute(0, 2, 1), visual_feats))
        
        visual_feats_ = self.conv(x).view(bs, -1, h * w)
        av_feats = torch.bmm(visual_feats_, attention.permute(0, 2, 1)).view(bs, -1, h, w)
        
        out = self.alpha * av_feats + x
          
        return out