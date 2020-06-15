import torch
from extract.detect.utils import iou_and_generalized_iou, xywh_to_x1x2y1y2
import numpy as np


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
    Basic block of detection network from:
    https://doi.org/10.1007/s00371-020-01831-7

    Based on Feature Pyramid Network core idea
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
    Modification of YOLOv3 network from
    https://doi.org/10.1007/s00371-020-01831-7

    Feature extraction network: DarkNet backbone with increased number
    of network layers of the first two residual blocks

    Detection network: Feature Pyramid Network
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

        y_small_shape = y_small.size()
        y_medium_shape = y_medium.size()
        y_large_shape = y_large.size()

        y_small = y_small.reshape(
            [y_small_shape[0], y_small_shape[2], y_small_shape[3], 3, -1]
        )
        y_medium = y_medium.reshape(
            [y_medium_shape[0], y_medium_shape[2], y_medium_shape[3], 3, -1]
        )
        y_large = y_large.reshape(
            [y_large_shape[0], y_large_shape[2], y_large_shape[3], 3, -1]
        )

        return y_small, y_medium, y_large


def get_abs_pred(y_pred, valid_anchors_wh, num_classes):
    """
    Taken and adopted from TensorFlow:
    https://github.com/ethanyanjiali/deep-vision/blob/6765dfcc36f209d2fded00fb11c6d1ae0da8e658/YOLO/tensorflow/yolov3.py#L238

    :param y_pred:
    :param valid_anchors_wh:
    :param num_classes:
    :return:
    """
    xy_rel_pred, wh_rel_pred, obj_pred, cls_pred = torch.split(y_pred, (2, 2, 1, num_classes), dim=-1)
    obj_pred = torch.sigmoid(obj_pred)
    cls_pred = torch.sigmoid(cls_pred)

    grid_size = y_pred.size()[1]
    # noinspection PyTypeChecker
    c_xy = torch.meshgrid(
        torch.arange(start=0, end=grid_size), torch.arange(start=0, end=grid_size)
    )
    c_xy = torch.stack(c_xy, dim=-1)
    c_xy = c_xy[:, :, None, :]

    b_xy = torch.sigmoid(xy_rel_pred) + c_xy.float()
    b_xy /= grid_size

    b_wh = torch.exp(wh_rel_pred) * valid_anchors_wh
    y_box = torch.cat([b_xy, b_wh], dim=-1)
    return y_box, obj_pred, cls_pred


def get_rel_true(y_true, valid_anchors_wh):
    """
    Taken and adopted from TensorFlow:
    https://github.com/ethanyanjiali/deep-vision/blob/master/YOLO/tensorflow/yolov3.py#L329

    :param y_true:
    :param valid_anchors_wh:
    :param num_classes:
    :return:
    """
    grid_size = y_true.size()[1]
    # noinspection PyTypeChecker
    c_xy = torch.meshgrid(
        torch.arange(start=0, end=grid_size), torch.arange(start=0, end=grid_size)
    )
    c_xy = torch.stack(c_xy, dim=-1)
    c_xy = c_xy[:, :, None, :]

    b_xy, b_wh = y_true[..., :2], y_true[..., 2:4]

    t_xy = b_xy * grid_size - c_xy
    t_wh = torch.log(b_wh / valid_anchors_wh)
    t_wh = torch.where(
        torch.isinf(t_wh) | torch.isnan(t_wh),
        torch.zeros_like(t_wh),
        t_wh
    )

    y_box = torch.cat([t_xy, t_wh], dim=-1)
    return y_box


class YoloLoss(torch.nn.Module):
    """
    Taken and adopted from TensorFlow:
    https://github.com/ethanyanjiali/deep-vision/blob/master/YOLO/tensorflow/yolov3.py#L352

    """
    def __init__(self, num_classes, valid_anchors_wh):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.valid_anchors_wh = valid_anchors_wh

        self.ignore_thresh = 0.5

        # Proportion of parts of loss
        # https://doi.org/10.1007/s00371-020-01831-7
        self.lambda_reg = 2
        self.lambda_cls = 0.5
        self.lambda_obj = 1
        self.lambda_no_obj = 0.5

    def forward(self, y_pred, y_true):
        xy_rel_pred, wh_rel_pred = y_pred[..., :2], y_pred[..., 2:4]
        xy_rel_pred = torch.sigmoid(xy_rel_pred)
        wh_rel_pred = torch.sigmoid(wh_rel_pred)

        xy_abs_true, wh_abs_true, obj_true, cls_true = torch.split(y_true, (2, 2, 1, self.num_classes), dim=-1)
        box_abs_true = torch.cat([xy_abs_true, wh_abs_true], dim=-1)
        box_abs_true = xywh_to_x1x2y1y2(box_abs_true)

        box_abs_pred, obj_pred, cls_pred = get_abs_pred(
            y_pred,
            self.valid_anchors_wh,
            self.num_classes
        )
        box_abs_pred = xywh_to_x1x2y1y2(box_abs_pred)

        box_rel_true = get_rel_true(y_true, self.valid_anchors_wh)
        xy_rel_true = box_rel_true[..., :2]
        wh_rel_true = box_rel_true[..., 2:4]

        xy_loss = self.__reg_loss(obj_true, xy_rel_pred, xy_rel_true)
        wh_loss = self.__reg_loss(obj_true, wh_rel_pred, wh_rel_true)
        cls_loss = self.__cls_loss(obj_true, cls_pred, cls_true)
        ignore_mask = self.__ignore_mask(box_abs_true, box_abs_pred)
        obj_loss, no_obj_loss = self.__obj_loss(ignore_mask, obj_pred, obj_true)
        return xy_loss + wh_loss + cls_loss + obj_loss

    def __ignore_mask(self, box_pred, box_true):
        box_true = box_true.reshape([box_true.size()[0], -1, 4])
        box_true, _ = torch.sort(box_true, dim=1, descending=True)
        box_true = box_true[:, :100, :]  # Take largest values

        box_pred_shape = box_pred.size()
        box_pred = box_pred.reshape([box_pred_shape[0], -1, 4])

        _, GIoU = iou_and_generalized_iou(box_pred, box_true)
        best_GIoU, _ = torch.max(GIoU, dim=-1)
        best_GIoU = best_GIoU.reshape(
            [box_pred_shape[0], box_pred_shape[1], box_pred_shape[2], box_pred_shape[3]]
        )
        ignore_mask = torch.where(
            best_GIoU < self.ignore_thresh, torch.tensor(0), torch.tensor(1)
        )
        ignore_mask = ignore_mask[:, :, :, :, None]
        return ignore_mask

    def __reg_loss(self, obj_true, xy_rel_pred, xy_rel_true):
        xy_loss = torch.sum((xy_rel_true - xy_rel_pred)**2, dim=-1)
        obj_true = torch.squeeze(obj_true, dim=-1)
        xy_loss = torch.sum(xy_loss * obj_true, dim=(1, 2, 3))
        return xy_loss * self.lambda_reg

    def __cls_loss(self, obj_true, cls_pred, cls_true):
        cls_loss = torch.nn.functional.binary_cross_entropy(cls_pred, cls_true)
        cls_loss = torch.sum(cls_loss * obj_true, dim=(1, 2, 3, 4))
        return cls_loss * self.lambda_cls

    def __obj_loss(self, ignore_mask, obj_pred, obj_true):
        entropy = torch.nn.functional.binary_cross_entropy(obj_pred, obj_true)
        obj_loss = torch.sum(obj_true * entropy, dim=(1, 2, 3, 4))
        no_obj_loss = torch.sum((1 - obj_true) * entropy * ignore_mask, dim=(1, 2, 3, 4))
        return obj_loss * self.lambda_obj, no_obj_loss * self.lambda_no_obj


if __name__ == '__main__':
    anchors_wh = torch.tensor(np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                           [59, 119], [116, 90], [156, 198], [373, 326]],
                          np.float32)) / 416
    t = torch.rand([1, 3, 416, 416])
    yolo = YoloFaceNetwork(num_anchors=3, num_classes=2)
    y_small, y_medium, y_large = yolo(t)
    y_true = torch.rand([1, 13, 13, 3, 7])
    loss = YoloLoss(num_classes=2, valid_anchors_wh=anchors_wh[:3])
    print(loss(y_small, y_true))
