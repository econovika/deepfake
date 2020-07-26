import torch


def iou_and_generalized_iou(box_pred, box_truth):
    """
    Calculate IoU and GIoU between each pair of box_pred and box_truth

    Generalized and adopted from TensorFlow:
    https://github.com/ethanyanjiali/deep-vision/blob/6765dfcc36f209d2fded00fb11c6d1ae0da8e658/YOLO/tensorflow/utils.py#L31

    Used algorithm to compute GIoU:
    https://giou.stanford.edu/

    :param box_pred: PyTorch tensor of predicted boxes (x1 y1 x2 y2 : x2 > x1 & y2 > y1), shape (n_batch, n, 4)
    :param box_truth: PyTorch tensor of ground-truth boxes (x1 y1 x2 y2 : x2 > x1 & y2 > y1), shape (n_batch, m, 4)
    :return: IoU and GIoU: PyTorch tensors, shapes (n_batch, n, m, 1)
    """
    # Broadcast tensors
    box_pred = box_pred[:, :, None, :]
    box_truth = box_truth[:, None, :, :]

    shape_pred, shape_truth = torch.tensor(box_pred.size()), torch.tensor(box_truth.size())
    shape_pred[-2], shape_truth[-3] = shape_truth[-2], shape_pred[-3]

    box_pred = box_pred.expand(shape_pred.tolist())
    box_truth = box_truth.expand(shape_truth.tolist())

    pred_left, pred_bottom, pred_right, pred_top = torch.split(box_pred, split_size_or_sections=1, dim=-1)
    truth_left, truth_bottom, truth_right, truth_top = torch.split(box_truth, split_size_or_sections=1, dim=-1)

    left, right = torch.max(pred_left, truth_left), torch.min(pred_right, truth_right)
    bottom, top = torch.max(pred_bottom, truth_bottom), torch.min(pred_top, truth_top)

    area_truth = (truth_right - truth_left) * (truth_top - truth_bottom)
    area_pred = (pred_right - pred_left) * (pred_top - pred_bottom)

    I = torch.clamp(right - left, min=0) * torch.clamp(top - bottom, min=0)  # Intersection area should be non-negative
    U = area_pred + area_truth - I

    left, right = torch.min(pred_left, truth_left), torch.max(pred_right, truth_right)
    bottom, top = torch.min(pred_bottom, truth_bottom), torch.max(pred_top, truth_top)

    area_convex = (right - left) * (top - bottom)  # Area of smallest convex hull which contains box_pred and box_truth
    IoU = I / (U + 1e-7)  # Avoid division by zero
    GIoU = IoU - (area_convex - U) / (area_convex + 1e-7)  # Avoid division by zero

    IoU = torch.squeeze(IoU, dim=-1)
    GIoU = torch.squeeze(GIoU, dim=-1)
    return IoU, GIoU


def xywh_to_x1y1x2y2(box):
    """
    Taken and adopted from TensorFlow:
    https://github.com/ethanyanjiali/deep-vision/blob/6765dfcc36f209d2fded00fb11c6d1ae0da8e658/YOLO/tensorflow/utils.py#L4
    """
    xy = box[..., 0:2]
    wh = box[..., 2:4]

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2

    y_box = torch.cat([x1y1, x2y2], dim=-1)
    return y_box


def find_best_anchors(boxes, anchors):
    boxes_wh = boxes[..., 2:4] - boxes[..., :2]
    boxes_wh = boxes_wh[:, :, None, :]
    boxes_wh_shape = torch.tensor(boxes_wh.size())
    boxes_wh_shape[2] = 9
    boxes_wh = boxes_wh.expand(boxes_wh_shape.tolist())

    intersection = min(boxes_wh[..., 0], anchors[..., 0]) * min(boxes_wh[..., 1], anchors[..., 1])
    box_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]

    iou = intersection / (box_area + anchor_area - intersection)
    return torch.argmax(iou, dim=-1)


def preprocess_boxes(boxes, best_anchors, anchors, valid_anchors, grid_size, num_classes, num_anchors):
    y_true = torch.zeros(grid_size, grid_size, num_anchors, (4 + 1 + num_classes))

    indices, updates = [], []

    for box, anchor in zip(boxes, best_anchors):
        obj = box[..., 4][:, None]
        cls = box[..., 5:]

        anchor_found = any(
            [torch.any(
                torch.any(torch.eq(valid_anchor, anchors[anchor]), dim=1)
            ).item() for valid_anchor in valid_anchors]
        )
        if anchor_found:
            adjusted_anchor_index = anchor % 3
            adjusted_anchor_index = adjusted_anchor_index.float()

            box_xy = (box[..., :2] + box[..., 2:4]) / 2
            box_wh = box[..., 2:4] - box[..., :2]

            grid_cell_xy = box_xy // (1 / grid_size)

            indices.extend(
                torch.cat([grid_cell_xy[:, 1][:, None],
                           grid_cell_xy[:, 0][:, None],
                           adjusted_anchor_index[:, None]], dim=-1)
            )
            updates.extend(torch.cat([box_xy, box_wh, obj, cls], dim=-1))
    indices = torch.stack(indices, dim=0).long()
    updates = torch.stack(updates, dim=0)
    y_true = y_true.tolist()
    for index, update in zip(indices, updates):
        y_true[index[0]][index[1]][index[2]] = update.tolist()
    return torch.tensor(y_true)
