import torch.nn
from basicsr.models.archs.edvr_arch import PCDAlignment


class ResBlock(torch.nn.Module):
    # Conv block with residual passthrough
    def __init__(self, features=64):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x):
        out = torch.nn.functional.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return x + out


class DownBlock(torch.nn.Module):
    # Strided convolution for down sampling
    def __init__(self, features=64):
        super(DownBlock, self).__init__()
        self.down_conv1 = torch.nn.Conv2d(
            features, features, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.down_conv11 = torch.nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.lrelu(self.down_conv11(self.lrelu(self.down_conv1(x))))


class Fusion(torch.nn.Module):
    # currently just averages - need to implement registration
    # with downsampled features and fusion with the registration
    def __init__(self, features=64):
        super(Fusion, self).__init__()

        # Add strided convolutions for the fusion for nonlocal features in reg
        self.down_conv2 = DownBlock(features=features)
        self.down_conv4 = DownBlock(features=features)

        # Using pyramidal cascading deformable convolution to align features
        self.PCDAlignment = PCDAlignment(num_feat=features, deformable_groups=8)

        # just use a single convolution to fuse the 2 features into 1
        self.fusion = torch.nn.Conv2d(
            2 * features, features, kernel_size=1, stride=1, bias=True
        )

    def forward(self, feat):
        # get 2x and 4x downsampled - the pyramidal representation required for PCD
        feat2 = self.down_conv2(feat)
        feat4 = self.down_conv4(feat2)

        # PCD requires list[tensor] of pyramid levels with shape (b,c,h,w)
        # so we need to rearrange as we have (b*t, c, h, w)
        # for each pyramid level
        batch_size = feat.shape[0] // 2

        # align the features - this aligns to the second argument
        # TODO rearrange slicing so memory is automatically contiguous
        aligned = self.PCDAlignment(
            [
                feat[::batch_size, :, :, :].contiguous(),
                feat2[
                    ::batch_size, :, :, :
                ].contiguous(),  # t - 1 features at 3 different resolutions (to be aligned)
                feat4[::batch_size, :, :, :].contiguous(),
            ],
            [
                feat[1::batch_size, :, :, :].contiguous(),
                feat2[
                    1::batch_size, :, :, :
                ].contiguous(),  # t + 1 features - (reference features)
                feat4[1::batch_size, :, :, :].contiguous(),
            ],
        )

        # concatenate the aligned and reference
        aligned = torch.cat([aligned, feat[1::batch_size, :, :, :]], dim=1)

        # and fuse with a simple conv
        fused = self.fusion(aligned)
        return fused


class InterpNet(torch.nn.Module):
    """
    Does interpolation in feature space without the spatial
    super-resolution step implemented in https://arxiv.org/pdf/2002.11616.pdf
    """

    def __init__(self):
        super(InterpNet, self).__init__()

        res_blocks = 5
        features = 64

        # Define the feature encoder
        self.entry_conv = torch.nn.Conv2d(
            1, features, kernel_size=3, padding=1, bias=True
        )
        self.encoder = torch.nn.ModuleList(
            [ResBlock(features=features) for x in range(res_blocks)]
        )

        # Registration and fusion
        self.fusion = Fusion(features=features)

        # Decoder output
        self.decoder = torch.nn.ModuleList(
            [ResBlock(features=features) for x in range(res_blocks)]
        )
        self.exit_conv = torch.nn.Conv2d(
            features, 1, kernel_size=3, padding=1, bias=True
        )

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # expects x as Batch, 2 frames, 1, height, width
        h, w = x.shape[-2:]

        # feature extraction
        feat = self.lrelu(self.entry_conv(x.view((-1, 1, h, w))))
        for block in self.encoder:
            feat = block(feat)

        # register and fuse
        out = self.fusion(feat)

        # decode
        for block in self.decoder:
            out = block(out)
        out = self.exit_conv(self.lrelu(out))

        return out


if __name__ == "__main__":
    device = "cuda"
    images = torch.rand(2, 2, 1, 512, 512)
    model = InterpNet().to(device)
    print(model(images.to(device)))
