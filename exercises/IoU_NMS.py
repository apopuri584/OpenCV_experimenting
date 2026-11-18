import cv2 as cv 
import numpy as np
import torch


#Implement IoU for two boxes from scratch, then NMS, and test on a list of overlapping boxes.
"""
Step by step: 
1. Get box 1 and box 2 x and y coordinates (x1, y1, x2, y2)
2. take maximum of first set of x and y coordiantes
3. take minimum of second set of x and y coordiantes
4. calculate the intersection: (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
5. calcualte union of boxes: abs((box1_x2 - box1_x1) * (box1_y1-box1_y2)) + abs((box2_x2 - box2_x1) * (box2_y1-box2_y2))
6. return the IOU score
"""
def contour_to_boxes(contour_list):
    bounding_boxes = []
    for contour in contour_list:
        area =  cv.contourArea(contour)
        if  110908.0 >  area > 100:
            for contour in contour_list:
                x1, y1, x2, y2 = cv.boundingRect(contour)
                bounding_boxes.append([0, area, x1, y1, x2, y2])

    return bounding_boxes


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



def nms(bboxes, iou_thresh, prob_treshold, box_format="corners"):
    # predictions = [[1(class), 0.9(probability), x1, y1, x2, y2]], [], []]
    

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_treshold]
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    bboxes_after_nms = []

    while bboxes: 
        chosen_box = bboxes.pop(0)
        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]), 
                box_format = box_format,
            ) < iou_thresh
                
        
            
        ]
        bboxes_after_nms.append(chosen_box)

    
    return bboxes_after_nms

def draw_boxes(bounding_boxes):
    for box in bounding_boxes:
        x, y, w, h = box[2:6]
        cv.rectangle(img, (x,y), (x+w, y+h),(0, 255, 0), 2)



img = cv.imread("../images/sunflower.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
ret, otsu_thresh = cv.threshold(gray, 125, 255, cv.THRESH_OTSU)


#1. Contours found using regular threshold
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


#2. contours found using otsu's threshold
otsu_contours, hierarchies = cv.findContours(otsu_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


all_bounding_boxes = contour_to_boxes(contours + otsu_contours)

all_bounding_boxes = nms(all_bounding_boxes, iou_thresh = 0.5, prob_treshold = 0, box_format="corners")

draw_boxes(all_bounding_boxes)



cv.imshow('Threshold', thresh)
cv.imshow('Otsu Threshold', otsu_thresh)
cv.imshow("Bounding boxes", img)



cv.waitKey(0)
cv.destroyAllWindows()

#print(f'{len(contours)} contours found!')


