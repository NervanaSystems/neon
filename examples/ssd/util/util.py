import numpy as np
from PIL import Image, ImageDraw

PIXEL_MEANS = np.array([[[104, 117, 123]]])
SCORE_IDX = 4


def convert_boxes_to_pixels(boxes, im_shape):
    """
    Converts box from normalized coordinates to pixels
    """
    boxes[:, 0] = boxes[:, 0] * im_shape[0]
    boxes[:, 1] = boxes[:, 1] * im_shape[1]
    boxes[:, 2] = boxes[:, 2] * im_shape[0]
    boxes[:, 3] = boxes[:, 3] * im_shape[1]

    return boxes


def prepare_image(img, im_shape, pixel_means=PIXEL_MEANS):
    """
    Given an image from the dataloader, undoes the transformations.
    """
    img = img.reshape((3, im_shape[0], im_shape[1]))
    # transpose from CHW to H W C
    img = img.transpose(1, 2, 0) + pixel_means
    # reorder color channel
    img = img[:, :, ::-1]

    img = img.astype('uint8')

    return Image.fromarray(img)


def draw_rectangle(draw, box, color, width):
    """
    Draws rectangle with provided color and line width
    """
    line = (box[0], box[1], box[0], box[3])
    draw.line(line, fill=color, width=width)

    line = (box[0], box[1], box[2], box[1])
    draw.line(line, fill=color, width=width)

    line = (box[0], box[3], box[2], box[3])
    draw.line(line, fill=color, width=width)

    line = (box[2], box[1], box[2], box[3])
    draw.line(line, fill=color, width=width)


def draw_boxes(img, gt_boxes, boxes):
    """
    Given an img, draws the gt_boxes as green,
    and the predicted boxes as red.
    """
    draw = ImageDraw.Draw(img)

    for box in gt_boxes:
        draw_rectangle(draw, box, 'green', 2)

    for box in boxes:
        box = np.array(box[:4], dtype=np.float32)
        draw_rectangle(draw, box, 'red', 2)

    return img


def plot_image(img, im_shape, gt_boxes, boxes, score_threshold):
    """
    Creates a PIL image object with the image, and the gt_boxes
    as green, and the predicted boxes as red. Only predicted
    boxes greater than score_threshold are kept.
    """
    img = prepare_image(img, im_shape)  # create PIL Image object

    gtb = convert_boxes_to_pixels(gt_boxes, im_shape)
    boxes = convert_boxes_to_pixels(boxes, im_shape)

    scores = boxes[:, SCORE_IDX]
    boxes = boxes[np.where(scores > score_threshold)[0], :]

    img = draw_boxes(img, gtb, boxes)

    return img


def JaccardOverlap(box1, box2):
    # ended up putting this inline above
    # check if there is any overlap
    # if (box1[:, 2] < box2[0]) or (box2[2] < box1[:, 0]) or \
    #    (box1[:, 3] < box2[1]) or (box2[3] < box1[:, 1]):
    #         return 0.0

    # get the intersection coords
    inter_xmin = np.maximum(box1[:, 0], box2[0])
    inter_xmax = np.minimum(box1[:, 2], box2[2])
    inter_ymin = np.maximum(box1[:, 1], box2[1])
    inter_ymax = np.minimum(box1[:, 3], box2[3])

    joint_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    joint_area += (box2[2] - box2[0]) * (box2[3] - box2[1])

    dx = (inter_xmax - inter_xmin)
    dy = (inter_ymax - inter_ymin)

    inter_area = np.sign(np.sign(dx) + np.sign(dy)) * dx * dy

    jac = inter_area / (joint_area - inter_area)
    return jac


def calculate_bb_overlap(rp, gt):

    R = rp.shape[0]
    G = gt.shape[0]

    overlaps = np.zeros((R, G))

    for ind, gt_box in enumerate(gt):
        overlap = JaccardOverlap(rp, gt_box)
        overlap[overlap <= 1.0e-6] = 0.0

        overlaps[:, ind] = overlap

    return overlaps
