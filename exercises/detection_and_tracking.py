import torch
import numpy as np
import random




"""
1. Assign ID's to previous frame boxes (chekc)
2. Check IOU scores for each frame in current frame boxes
3. Assign ID's based on which IOU score is highest 
4. Repeat for each frame
"""


def make_crossing_frames(n_frames=40, seed=0, speed=10, y_gap=0,
                         jitter=2, miss_prob=0.0, size=60):
    """Two boxes moving toward each other along the same row, then crossing.
    speed: pixels per frame; y_gap: vertical offset between the two boxes
    jitter: random noise in pixels (simulates a noisy detector)
    miss_prob: chance a box is missing in a frame (simulates missed detections)"""
    rng = random.Random(seed)
    frames = []
    for t in range(n_frames):
        objs = [
            (0, 20 + speed * t, 100),                                   # moves right
            (1, 20 + speed * (n_frames - 1) - speed * t, 100 + y_gap),  # moves left
        ]
        frame = []
        for true_id, x, y in objs:
            if rng.random() < miss_prob:
                continue
            x += rng.randint(-jitter, jitter)
            y += rng.randint(-jitter, jitter)
            frame.append((true_id, [x, y, x + size, y + size]))
        rng.shuffle(frame)          # so list order gives no hint
        frames.append(frame)
    return frames


def iou(box1, box2, box_format = "corners"):
    #boxes_preds shape is (N, 4) where N is the number of boxes, 4 is (x1, y1, x2, y2)
    #boxes_labels shape is (N, 4) where N is the number of boxes
    box_pred = torch.tensor(box1)
    box_labels = torch.tensor(box2)
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

frames = make_crossing_frames()
# print(frames[5])        # one instant: a list of tuples, one per object
# print(frames[5][0][1])     # the first tuple in that instant: one box
# print(len(frames[5]))

# for frame in frames:
#     print(frame)

# ID assigning below
for box_index, (_, box) in enumerate(frames[0]):
    frames[0][box_index] = (box_index, box)

next_id = len(frames[0])

for frame_index in range(len(frames) - 1):
    previous_frame = frames[frame_index]
    current_frame = frames[frame_index + 1]
    matches = []

    for previous_index, (previous_id, previous_box) in enumerate(previous_frame):
        for current_index, (_, current_box) in enumerate(current_frame):
            overlap = iou(previous_box, current_box, box_format="corners").item()
            if overlap > 0.5:
                matches.append((overlap, previous_index, current_index, previous_id))

    matches.sort(reverse=True)
    matched_previous = set()
    matched_current = set()
    current_ids = [None] * len(current_frame)

    for _, previous_index, current_index, previous_id in matches:
        if previous_index not in matched_previous and current_index not in matched_current:
            current_ids[current_index] = previous_id
            matched_previous.add(previous_index)
            matched_current.add(current_index)

    for current_index, (_, box) in enumerate(current_frame):
        if current_ids[current_index] is None:
            current_ids[current_index] = next_id
            next_id += 1
        current_frame[current_index] = (current_ids[current_index], box)

            
        









