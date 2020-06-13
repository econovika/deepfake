import torch


class Conv2dBlock(torch.nn.Module):
    """
    Wrapper to standard torch.nn.Conv2d with automatic padding
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(Conv2dBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.padding = kernel_size // 2  # Padding size to keep input image dimensions

        self.model = self.__build_conv_block()

    def __build_conv_block(self):
        block = [
            torch.nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding),
            torch.nn.BatchNorm2d(num_features=self.out_channels),
            torch.nn.LeakyReLU(negative_slope=0.1)
        ]
        return torch.nn.Sequential(*block)

    def forward(self, x):
        x = self.model(x)
        return x


class ResBlock(torch.nn.Module):
    """
    DarkNet residual block
    """
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self._channels = self.in_channels * 2

        self.block = self.__build_res_block()

    def __build_res_block(self):
        block = [
            Conv2dBlock(in_channels=self.in_channels,
                        out_channels=self._channels),
            Conv2dBlock(in_channels=self._channels,
                        out_channels=self.in_channels,
                        kernel_size=3)
        ]
        return torch.nn.Sequential(*block)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x


class FeatureExtractionBlock(torch.nn.Module):
    """
    Basic block of feature extraction network from:
    https://doi.org/10.1007/s00371-020-01831-7

    ... --> Conv2dBlock --> ResBlock * num_layers --> ...
    """
    def __init__(self, in_channels, out_channels, num_layers):
        super(FeatureExtractionBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_layers = num_layers

        self.block = self.__build_block()

    def __build_block(self):
        block = [
            Conv2dBlock(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=2)
        ]
        for _ in range(self.num_layers):
            block.append(
                ResBlock(in_channels=self.out_channels)
            )
        return torch.nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DetectionBlock(torch.nn.Module):
    """

    """
    def __init__(self, in_channels, _channels, out_channels, mode='up_conv'):
        assert mode in ('up_conv', 'conv')
        super(DetectionBlock, self).__init__()
        self.in_channels, self._channels = in_channels, _channels
        self.out_channels = out_channels
        self.mode = mode
        # Swap channels if mode is 'conv'
        if self.mode == 'conv':
            self.in_channels, self._channels = self._channels, self.in_channels

        self.up_conv_block = self.__build_up_conv_block()
        self.conv_block, self.pred_conv_block = self.__build_conv_block()

    def __build_up_conv_block(self):
        block = [
            Conv2dBlock(in_channels=self.in_channels,
                        out_channels=self._channels),
            torch.nn.Upsample(mode='nearest', scale_factor=2)
        ]
        return torch.nn.Sequential(*block)

    def __build_conv_block(self):
        block = []
        for _ in range(3):
            block.append(
                Conv2dBlock(in_channels=self._channels,
                            out_channels=self.in_channels)
            )
            block.append(
                Conv2dBlock(in_channels=self.in_channels,
                            out_channels=self._channels,
                            kernel_size=3)
            )
        pred_block = block.copy()  # Detection block of particular scale
        pred_block.append(
            Conv2dBlock(in_channels=self._channels,
                        out_channels=self.out_channels)
        )
        return torch.nn.Sequential(*block), torch.nn.Sequential(*pred_block)

    def forward(self, x, x_backbone=None):
        x = self.up_conv_block(x) if self.mode == 'up_conv' else x
        x += x_backbone if x_backbone is not None else x  # Elementwise addition ???
        y = self.pred_conv_block(x)
        x = self.conv_block(x)
        return x, y


class YoloFaceNetwork(torch.nn.Module):
    """

    """
    def __init__(self, num_anchors, num_classes):
        super(YoloFaceNetwork, self).__init__()
        # 4: location offset against the anchor box: tx, ty, tw, th
        # 1: whether object is contained in box
        self.out_channels = num_anchors * (4 + 1 + num_classes)

        self.extraction_model = self.__build_feature_extraction_network()
        self.detection_model = self.__build_detection_network()

    @staticmethod
    def __build_feature_extraction_network():
        in_channels, out_channels = 3, 32
        model = [
            Conv2dBlock(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3)
        ]
        in_channels = out_channels
        for out_channels, num_layers in [(64, 4), (128, 8), (256, 8), (512, 8), (1024, 4)]:
            model.append(
                FeatureExtractionBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       num_layers=num_layers)
            )
            in_channels = out_channels
        return model

    def __build_detection_network(self):
        model = []
        for _channels, mode in [(512, 'conv'), (512, 'up_conv'), (256, 'up_conv')]:
            in_channels = _channels * 2
            model.append(
                DetectionBlock(in_channels=in_channels,
                               _channels=_channels,
                               out_channels=self.out_channels,
                               mode=mode)
            )
        return model

    def forward(self, x):
        for block in self.extraction_model[:3]:
            x = block(x)
        x_large = self.extraction_model[3](x)
        x_medium = self.extraction_model[4](x_large)
        x_small = self.extraction_model[5](x_medium)

        x_small, y_small = self.detection_model[0](x_small)
        x_medium, y_medium = self.detection_model[1](x_small, x_medium)
        _, y_large = self.detection_model[2](x_medium, x_large)
        return y_small, y_medium, y_large


if __name__ == '__main__':
    t = torch.rand(1, 3, 416, 416)
    cls = YoloFaceNetwork(3, 2)
    print(cls(t))
