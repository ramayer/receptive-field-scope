"""A tiny convolutional autoencoder (stride 8, quantised latents) that trains in about a minute on CPU.

Used by notebooks/lesion_demo.ipynb as a fallback when pretrained CompressAI weights cannot be downloaded. It is a TOY:
trained on a handful of sample images, so conclusions about it are about the method, not about real tokenizers."""
import torch
import torch.nn as nn


class TinyAE(nn.Module):
    def __init__(self, latent=48, width=48):
        super().__init__()
        act = nn.GELU
        self.enc = nn.Sequential(nn.Conv2d(3, width, 5, 2, 2), act(), nn.Conv2d(width, width, 5, 2, 2), act(),
                                 nn.Conv2d(width, latent, 5, 2, 2))
        self.dec = nn.Sequential(nn.ConvTranspose2d(latent, width, 5, 2, 2, output_padding=1), act(),
                                 nn.ConvTranspose2d(width, width, 5, 2, 2, output_padding=1), act(),
                                 nn.ConvTranspose2d(width, 3, 5, 2, 2, output_padding=1))

    def forward(self, x):
        y = self.enc(x)
        yq = y + (torch.rand_like(y) - 0.5) if self.training else torch.round(y)     # noise proxy for rounding
        return self.dec(yq), y


def train_tiny_autoencoder(images, steps=900, crop=64, batch=32, lam=2e-3, seed=0, device="cpu", verbose=True):
    """images: list of [1,3,H,W] tensors in [0,1]. Loss = MSE + lam * mean|y| (crude rate proxy)."""
    torch.manual_seed(seed)
    m = TinyAE().to(device)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    m.train()
    for s in range(steps):
        xs = []
        for _ in range(batch):
            im = images[torch.randint(len(images), (1,)).item()]
            i = torch.randint(im.shape[2] - crop + 1, (1,)).item()
            j = torch.randint(im.shape[3] - crop + 1, (1,)).item()
            xs.append(im[:, :, i:i + crop, j:j + crop])
        x = torch.cat(xs).to(device)
        xh, y = m(x)
        loss = (xh - x).pow(2).mean() + lam * y.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and s % 150 == 0:
            print(f"step {s:4d}  mse {(xh - x).pow(2).mean().item():.5f}  |y| {y.abs().mean().item():.3f}")
    return m.eval()
