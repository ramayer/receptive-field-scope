import torch
import torch.nn as nn

from receptive_field_scope import cut_profile, make_mask, margin_needed
from toys import ConvTokenizer
from test_lesion import LocalDec

torch.manual_seed(0)


class RegisterTokenizer(ConvTokenizer):
    """ConvTokenizer plus a 'register': the top-left token's first channel is broadcast to every token (global state)."""
    def forward(self, x):
        z = super().forward(x)
        return z + 3.0 * z[:, :1, :1, :1]


def test_local_encoder_margin_matches_structure_crop_and_replace():
    m = ConvTokenizer(5).eval()                                   # support 45px one-sided -> 6 cells of 8px
    x = torch.randn(1, 3, 256, 256)
    for mode, kw in [("crop", dict(cut=(0, 0, 0, 8))), ("replace", dict(cut=(0, 0, 0, 8)))]:
        if mode == "replace":
            mask = torch.zeros(32, 32, dtype=torch.bool); mask[:, 24:] = True
            p = cut_profile(m, x, mode="replace", replace_mask=mask, maxd=12)
        else:
            p = cut_profile(m, x, **kw, mode="crop", maxd=12)
        need = margin_needed(p, rel_tol=1e-10)
        assert need is not None and 3 <= need <= 7, (mode, need, p["rel_error"])


def test_register_in_cut_band_leaks_far_but_other_band_does_not():
    m = RegisterTokenizer(2).eval()
    x = torch.randn(1, 3, 256, 256)
    corner = torch.zeros(32, 32, dtype=torch.bool); corner[:3, :3] = True          # contains the register
    other = torch.zeros(32, 32, dtype=torch.bool); other[-3:, -3:] = True
    p_reg = cut_profile(m, x, mode="replace", replace_mask=corner, maxd=20)
    p_oth = cut_profile(m, x, mode="replace", replace_mask=other, maxd=20)
    assert margin_needed(p_reg, 1e-10) is None                    # error never settles: global state
    assert margin_needed(p_oth, 1e-10) is not None                # an ordinary corner: local, settles


def test_decoder_cut_profile_and_register():
    dec = LocalDec().eval()                                         # reach: 1 cell
    y = torch.randn(1, 8, 16, 16)
    pc = cut_profile(dec, y, (0, 0, 0, 3), mode="crop", maxd=8)
    pr = cut_profile(dec, y, (0, 0, 0, 3), mode="replace", maxd=8)
    assert margin_needed(pc, 1e-10) is not None and margin_needed(pc, 1e-10) <= 2
    assert margin_needed(pr, 1e-10) is not None and margin_needed(pr, 1e-10) <= 2
    reg = LocalDec(register=5.0).eval()                              # token (0,0) channel 0 broadcast everywhere
    corner = torch.zeros(16, 16, dtype=torch.bool); corner[:2, :2] = True
    assert margin_needed(cut_profile(reg, y, mode="replace", replace_mask=corner, maxd=10), 1e-10) is None
