from extract.detect.utils import iou_and_generalized_iou, xywh_to_x1x2y1y2
from extract.detect.yolo_face import compute_abs_boxes
from torch import nonzero, argmax, cat, stack


class Process:
    def __init__(
            self, y_pred,
            num_classes, valid_anchors_wh,
            GIoU_thresh, score_thresh, max_detection=100
    ):
        self.GIoU_thresh = GIoU_thresh
        self.score_thresh = score_thresh
        self.max_detection = max_detection

        boxes, objects_scores, classes_probs = compute_abs_boxes(
            num_classes=num_classes,
            valid_anchors_wh=valid_anchors_wh,
            y_pred=y_pred
        )
        batch_size = boxes.size()[0]

        boxes = xywh_to_x1x2y1y2(boxes.reshape([batch_size, -1, 4]))
        objects_scores = objects_scores.reshape([batch_size, -1, 1])
        classes_probs = classes_probs.reshape([batch_size, -1, num_classes])

        self.raw_boxes, self.raw_objects_scores, self.raw_classes_probs = boxes, objects_scores, classes_probs

    def __call__(self, *args, **kwargs):
        return self.__filter_candidate_pred(
            cat([self.raw_boxes, self.raw_objects_scores, self.raw_classes_probs], dim=2)
        )

    def __filter_candidate_pred(self, processed_pred):
        # Filter out predictions with low than self.score_thresh objects prediction scores
        mask = (processed_pred[..., 4] >= self.score_thresh).squeeze(dim=0)
        processed_pred = processed_pred[:, nonzero(mask, as_tuple=False), :].squeeze(dim=2)

        best_preds = []
        num_detected = 0
        while processed_pred.size()[1] > 0 and num_detected < self.max_detection:
            num_detected += 1
            # Each step find prediction with highest score
            best_pred_idx = argmax(processed_pred[..., 4], dim=1)
            best_pred = processed_pred[:, best_pred_idx, :]
            best_preds.append(best_pred)

            # Filter out all predictions which overlap the best found on this step
            _, GIoU = iou_and_generalized_iou(best_pred[..., :4], processed_pred[..., :4])
            mask = (GIoU[0] <= self.GIoU_thresh).squeeze(dim=0)
            processed_pred = processed_pred[:, nonzero(mask, as_tuple=False), :].squeeze(dim=2)
        return stack(best_preds, dim=2).squeeze(dim=0)
