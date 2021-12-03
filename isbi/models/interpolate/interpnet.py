import torch.nn


class ResBlock(torch.nn.Module):
    # Conv block with residual passthrough
    def __init__(self, features=64):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            features, features, kernel_size=3, padding="same", bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            features, features, kernel_size=3, padding="same", bias=True
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
    def __init__(self):
        super(Fusion, self).__init__()
        self.test_conv = torch.nn.Conv2d(64, 64, 1, 1, bias=True)

    def forward(
        self,
        feat1,
        feat2,
        feat4,
    ):
        # currently just returns the first of the frames
        return self.test_conv(feat1)[::2, ...]


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

        # Add strided convolutions for the fusion for nonlocal features in reg
        self.down_conv2 = DownBlock(features=features)
        self.down_conv4 = DownBlock(features=features)

        # Registration and fusion
        # TODO - currently just a passthrough (this is the main bit!!)
        self.fusion = Fusion()

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
        feat1 = self.lrelu(self.entry_conv(x.view((-1, 1, h, w))))
        for block in self.encoder:
            feat1 = block(feat1)

        # get 2x and 4x downsampled
        feat2 = self.down_conv2(feat1)
        feat4 = self.down_conv4(feat2)

        print(feat1.shape)
        # register and fuse
        out = self.fusion(feat1, feat2, feat4)

        print(out.shape)

        # decode
        for block in self.decoder:
            out = block(out)
        out = self.exit_conv(self.lrelu(out))

        return out


if __name__ == "__main__":
    images = torch.rand(2, 2, 1, 512, 512)
    model = InterpNet()
    print(model(images))
