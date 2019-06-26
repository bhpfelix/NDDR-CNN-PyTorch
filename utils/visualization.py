from PIL import Image
import numpy as np


label_colours = [(178, 45, 45), (153, 115, 115), (64, 36, 32), (255, 68, 0), (89, 24, 0), (191, 121, 96), (191, 102, 0),
                 (76, 41, 0), (153, 115, 38), (102, 94, 77), (242, 194, 0), (191, 188, 143), (226, 242, 0),
                 (119, 128, 0), (59, 64, 0), (105, 191, 48), (81, 128, 64), (0, 255, 0), (0, 51, 7), (191, 255, 208),
                 (96, 128, 113), (0, 204, 136), (13, 51, 43), (0, 191, 179), (0, 204, 255), (29, 98, 115), (0, 34, 51),
                 (163, 199, 217), (0, 136, 255), (41, 108, 166), (32, 57, 128), (0, 22, 166), (77, 80, 102),
                 (119, 54, 217), (41, 0, 77), (222, 182, 242), (103, 57, 115), (247, 128, 255), (191, 0, 153),
                 (128, 96, 117), (127, 0, 68), (229, 0, 92), (76, 0, 31), (255, 128, 179), (242, 182, 198)]


def process_image(image):
    mean = np.array([122.67891434, 116.66876762, 104.00698793])
    std = np.array([1., 1., 1.])
    image = image.cpu().numpy() * std[:, None, None] + mean[:, None, None]
    return image.astype(np.uint8)


def process_seg_label(pred, gt, num_classes=40):
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    h, w = gt.shape
    pred_img = Image.new('RGB', (w, h), (255, 255, 255))  # unlabeled part is white (255, 255, 255)
    gt_img = Image.new('RGB', (w, h), (255, 255, 255))
    pred_pixels = pred_img.load()
    gt_pixels = gt_img.load()
    for j_, j in enumerate(gt):
        for k_, k in enumerate(j):
            if k < num_classes:
                gt_pixels[k_, j_] = label_colours[k]
                pred_pixels[k_, j_] = label_colours[pred[j_, k_]]
    return np.array(pred_img).transpose([2, 0, 1]), np.array(gt_img).transpose([2, 0, 1])


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=0, keepdims=True)


def process_normal_label(pred, gt, ignore_label):
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    mask = gt != ignore_label
    _, h, w = gt.shape
    pred = normalize(pred.reshape(3, -1)).reshape(3, h, w) * mask + (1 - mask)
    gt = normalize(gt.reshape(3, -1)).reshape(3, h, w) * mask + (1 - mask)
    return pred, gt
