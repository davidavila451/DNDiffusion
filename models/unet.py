from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel(
    sample_size=96,
    in_channels=4,
    out_channels=4,
    layers_per_block=3,
    block_out_channels=(512, 1024, 2048, 2048),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=512,
)