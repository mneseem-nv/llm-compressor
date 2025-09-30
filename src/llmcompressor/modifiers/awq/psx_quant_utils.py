from psx_formats.utils import _C  # where nvfp4_clippy lives

def psx_nvfp4(x):
    amax = torch.max(torch.abs(x.to(torch.float)))
    return _C.nvfp4_clippy(x, amax)

def psx_mxfp4(x):
    return _C.mxfp4_clippy(x)