import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """3x3 stem conv -> patchify (stride 8) -> `hops` token-space 3x3 convs. Support width = 8 + 2*8*hops + 2."""
    def __init__(self, hops=5):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.pat = nn.Conv2d(16, 32, 8, stride=8)
        self.body = nn.Sequential(*[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU()) for _ in range(hops)])

    def forward(self, x):
        return self.body(self.pat(self.stem(x)))


class GlobalMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1)
        self.g = nn.Conv2d(8, 8, 1)

    def forward(self, x):
        h = self.c(x)
        return h + self.g(h.mean((2, 3), keepdim=True))


class BlockedMix(nn.Module):
    """Each cell sees its own `b`x`b` block plus a `h`-cell halo (mean-pool stand-in for attention), behind a
    zero-init residual gate. Worst-case one-sided reach = b - 1 + h cells."""
    def __init__(self, b=4, h=1, ch=8):
        super().__init__()
        self.b, self.h = b, h
        self.v = nn.Conv2d(ch, ch, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        b, h = self.b, self.h
        xp = nn.functional.pad(x, (h, h, h, h))
        out = x.clone()
        ctx = torch.zeros_like(x)
        for by in range(0, H, b):
            for bx in range(0, W, b):
                ctx[:, :, by:by + b, bx:bx + b] = xp[:, :, by:by + b + 2 * h, bx:bx + b + 2 * h].mean((2, 3), keepdim=True)
        return out + self.gate * self.v(ctx)
