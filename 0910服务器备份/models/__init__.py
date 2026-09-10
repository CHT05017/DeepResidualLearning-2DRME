import torch
from models.UNet import UNet
from models.R2Net import R2Net
from models.RadioUNet import RadioUNet
from utils.overheads import count_params, macs_flops

RadioLAM_UNET_INPUT_CHANNEL = 6
R2Net_UNET_INPUT_CHANNEL = 4


def make_UNet(
        cfg,
        Logger
):
    """
    ### Functions:
    - Make RadioLAM's single generator (2026 JSAC)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        # [x_t, physics_map, height_map, building, terrain]
        input_shape=(RadioLAM_UNET_INPUT_CHANNEL, 128, 128), # SpectrumNet
        cfg=cfg
    )

    model = model.to(device=device)
    total, trainable = count_params(model=model)

    macs, flops = macs_flops(model=model, device=device, cfg=cfg, C=RadioLAM_UNET_INPUT_CHANNEL, is_diffusion=True)

    Logger.info(f"RadioLAM (2026 JSAC) has been successfully loaded.")
    Logger.info(f"Model device: {device}")
    Logger.info(f"Model total parameters: {total / 1e6:.2f} M")
    Logger.info(f"Model trainable parameters: {trainable / 1e6:.2f} M")
    Logger.info(f"Model MACs: {macs}; FLOPs: {flops}.")

    return model

def make_R2Net(
        cfg,
        Logger,
):
    """
    ### Functions:
    - make R2Net (2026 TVT)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = R2Net(
        in_channel=R2Net_UNET_INPUT_CHANNEL,
        out_channel=1
    )

    model = model.to(device=device)
    total, trainable = count_params(model=model)

    macs, flops = macs_flops(model=model, device=device, cfg=cfg, C=R2Net_UNET_INPUT_CHANNEL, is_diffusion=False)

    Logger.info(f"R2Net (2026 TVT) has been successfully loaded.")
    Logger.info(f"Model device: {device}")
    Logger.info(f"Model total parameters: {total / 1e6:.2f} M")
    Logger.info(f"Model trainable parameters: {trainable / 1e6:.2f} M")
    Logger.info(f"Model MACs: {macs}; FLOPs: {flops}.")

    return model

def make_RadioUNet(
        cfg,
        Logger,
        radiounet_phase="firstU"
):
    """
    ### Functions:
    - Make RadioUNet (2021 TWC)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RadioUNet(
        inputs=RadioLAM_UNET_INPUT_CHANNEL,
        phase=radiounet_phase
    )

    model = model.to(device=device)
    total, trainable = count_params(model=model)

    macs, flops = macs_flops(model=model, device=device, cfg=cfg, C=R2Net_UNET_INPUT_CHANNEL, is_diffusion=False)

    Logger.info(f"RadioUNet (2021 TWC) has been successfully loaded.")
    Logger.info(f"Model device: {device}")
    Logger.info(f"Model total parameters: {total / 1e6:.2f} M")
    Logger.info(f"Model trainable parameters: {trainable / 1e6:.2f} M")
    Logger.info(f"Model MACs: {macs}; FLOPs: {flops}.")

    return model

