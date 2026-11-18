import torch
#general implementations
def iou(box_pred, box_labels, box_format = "midpoint"):
    #boxes_preds shape is (N, 4) where N is the number of boxes, 4 is (x1, y1, x2, y2)
    #boxes_labels shape is (N, 4) where N is the number of boxes
    if box_format == "midpoint":
        box1_x1 = box_pred[..., 0:1] - box_pred[..., 2:3] / 2
        box1_y1 = box_pred[..., 1:2] - box_pred[..., 3:4] / 2
        box1_x2 = box_pred[..., 0:1] + box_pred[..., 2:3] / 2
        box1_y2 = box_pred[..., 1:2] + box_pred[..., 3:4] / 2

        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4] / 2


    if box_format == "corners":
        
        box1_x1 = box_pred[...,0:1]
        box1_y1 = box_pred[...,1:2]
        box1_x2 = box_pred[...,2:3]
        box1_y2 = box_pred[...,3:4]

        box2_x1 = box_labels[...,0:1]
        box2_y1 = box_labels[...,1:2]
        box2_x2 = box_labels[...,2:3]
        box2_y2 = box_labels[...,3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #.clamp(0) is for the case when they do not inersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1-box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1-box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)






