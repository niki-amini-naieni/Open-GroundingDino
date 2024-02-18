# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, os
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"{boxes1}"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"{boxes2}"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def boxes_to_masks(boxes, W, H):
    """
    Converts boxes to binary masks for Swin transformer layers at 8, 16, 32 patch size resolutions.

    The boxes should be in format [N, cx, cy, w, h] where N is the number of boxes.
    W is the image width, H is the image height.

    Output masks are returned as an N-length list of dictionaries with keys 8, 16, 32, 64 with values of dims (ceil(H / 8), ceil(W / 8)), (ceil(H / 16), ceil(W / 16), (ceil(H / 32), ceil(W / 32)), (ceil(H / 64), ceil(W / 64)) 
    """
    boxes = box_cxcywh_to_xyxy(boxes)
    y = torch.arange(0, H, dtype=torch.float) / H
    x = torch.arange(0, W, dtype=torch.float) / W
    y, x = torch.meshgrid(y, x)
    masks = []
    for box in boxes:
        mask = x >= box[0]
        mask = mask * (x <= box[2])
        mask = mask * (y >= box[1])
        mask = mask * (y <= box[3])
        mask_dict = {}
        mask_dict[8] = torch.nn.MaxPool2d((8, 8), ceil_mode=True)(mask.float().unsqueeze(0)).squeeze()
        mask_dict[16] = torch.nn.MaxPool2d((16, 16), ceil_mode=True)(mask.float().unsqueeze(0)).squeeze()
        mask_dict[32] = torch.nn.MaxPool2d((32, 32), ceil_mode=True)(mask.float().unsqueeze(0)).squeeze()
        mask_dict[64] = torch.nn.MaxPool2d((64, 64), ceil_mode=True)(mask.float().unsqueeze(0)).squeeze()
        masks.append(mask_dict)

    return masks

def pad_boxes_to_max(boxes):
    """
    Pads box masks so that all masks in [boxes] have the same width and height, namely, the max width in [boxes] and the max height in [boxes]. Padding is applied to the right and bottom of the boxes.

    boxes should be a list of stacked box masks. Each stack should have masks of the same height and width, but each stack in the list may have different heights and widths to be resolved with [pad_boxes_to_max]. If a stack is empty (i.e., has no boxes), a single mask of zeros is created for that stack.

    [boxes]: a list where each element is a stack of box masks of dim N x H x W, with N being the number of boxes in the image of size H x W, or of dim 1 with 0 length for no boxes.
    """
    max_h = 0
    max_w = 0
    for box in boxes:
        if box.shape[0] == 0:
            continue
        if box.shape[1] > max_h:
            max_h = box.shape[1]
        if box.shape[2] > max_w:
            max_w = box.shape[2]
    new_boxes = []
    for box in boxes:
        if box.shape[0] > 0:
            new_boxes.append(torch.nn.functional.pad(box, (0, max_w - box.shape[2], 0, max_h - box.shape[1])))
        else:
            new_boxes.append(torch.zeros((max_h, max_w)))

    return new_boxes


def pad_boxes(boxes, H, W):
    """
    Pads box masks so that all masks in [boxes] have the same width and height, namely, the max width in [boxes] and the max height in [boxes]. Padding is applied to the right and bottom of the boxes.

    boxes should be a list of stacked box masks. Each stack should have masks of the same height and width, but each stack in the list may have different heights and widths to be resolved with [pad_boxes_to_max]. If a stack is empty (i.e., has no boxes), a single mask of zeros is created for that stack.

    [boxes]: a list where each element is a stack of box masks of dim N x H x W, with N being the number of boxes in the image of size H x W, or of dim 1 with 0 length for no boxes.
    """
    new_boxes = []
    for box in boxes:
        if box.shape[0] > 0:
            new_boxes.append(torch.nn.functional.pad(box, (0, W - box.shape[2], 0, H - box.shape[1])))
        else:
            new_boxes.append(torch.zeros((H, W)))

    return new_boxes
 


if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb; ipdb.set_trace()
