import torch
import torch.nn as nn
import math

# Maps integer timestep → vector, same idea as positional encoding in transformers.
# Low frequencies change slowly across timesteps, high frequencies change fast,
# giving the network a rich representation of "where we are" in the noise schedule.

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2 # we divide the dim by 2 as half embeddings will be covered by sine and half embeddings will be covered by cosine
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half) # this will give us decaying values from 1 to exp(-log(1000))
        args = t[:, None].float() * freqs[None, :] # t (B, 1) * freqs (1,64) = (B,64)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (B, 128)


# The basic building block: two convolutions with a timestep injection in between.
# GroupNorm + SiLU as specified in the assignment.

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # group norm is preferred over batch norm in diffusion models as batch size is usually smaller. Group norm works by splitting channels into groups, 8 groups over here and normalizing by each channels own mean and var.
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)  # project time embedding to channel dim

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU() # This is defined as x * sigma (x), it is more expressive than ReLU as there is no hard clip anywhere, and allows negative values through

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add timestep info: broadcast (B, out_ch) → (B, out_ch, 1, 1)
        h = h + self.time_mlp(t_emb)[:, :, None, None] # same time embedding gets broadcasted to every position.

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)



class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, channels=(32, 64, 128), time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        self.down1 = ResBlock(channels[0], channels[0], time_dim)
        self.downsample1 = Downsample(channels[0])       

        self.down2 = ResBlock(channels[0], channels[1], time_dim)
        self.downsample2 = Downsample(channels[1])      

        self.bottleneck = ResBlock(channels[1], channels[2], time_dim)
        self.upsample2 = Upsample(channels[2])      
        self.up2 = ResBlock(channels[2] + channels[1], channels[1], time_dim)

        self.upsample1 = Upsample(channels[1])         
        self.up1 = ResBlock(channels[1] + channels[0], channels[0], time_dim) 
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        """
        x: (B, 1, 28, 28) noisy image
        t: (B,) integer timesteps
        returns: (B, 1, 28, 28) predicted noise
        """
        t_emb = self.time_embed(t)

        # Entry
        h = self.conv_in(x)                    

        # Encoder
        h1 = self.down1(h, t_emb)              
        h = self.downsample1(h1)               

        h2 = self.down2(h, t_emb)             
        h = self.downsample2(h2)              

        # Bottleneck
        h = self.bottleneck(h, t_emb)         

        # Decoder
        h = self.upsample2(h)                
        h = torch.cat([h, h2], dim=1)          
        h = self.up2(h, t_emb)                 

        h = self.upsample1(h)                  
        h = torch.cat([h, h1], dim=1)          
        h = self.up1(h, t_emb)               
        return self.conv_out(h)              
