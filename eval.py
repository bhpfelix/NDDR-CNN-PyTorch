import os
import argparse
from configs import cfg
from tqdm import tqdm
import numpy as np

import torch

from data.loader import MultiTaskDataset

from models.nddr_net import NDDRNet
from models.vgg16_lfov import DeepLabLargeFOV

from utils.metrics import compute_hist, compute_angle


def main():
    parser = argparse.ArgumentParser(description="PyTorch NDDR Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # load the data
    test_loader = torch.utils.data.DataLoader(
        MultiTaskDataset(
            data_dir=cfg.DATA_DIR,
            data_list_1=cfg.TEST.DATA_LIST_1,
            data_list_2=cfg.TEST.DATA_LIST_2,
            output_size=cfg.TEST.OUTPUT_SIZE,
            random_scale=cfg.TEST.RANDOM_SCALE,
            random_mirror=cfg.TEST.RANDOM_MIRROR,
            random_crop=cfg.TEST.RANDOM_CROP,
            ignore_label=cfg.IGNORE_LABEL,
        ),
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

    net1 = DeepLabLargeFOV(3, cfg.MODEL.NET1_CLASSES, weights='')
    net2 = DeepLabLargeFOV(3, cfg.MODEL.NET2_CLASSES, weights='')
    model = NDDRNet(net1, net2)
    ckpt_path = os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME, 'ckpt-%s.pth' % str(cfg.TEST.CKPT_ID).zfill(5))
    print("Evaluating Checkpoint at %s" % ckpt_path)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    with torch.no_grad():
        total_hist = np.zeros((cfg.MODEL.NET1_CLASSES, cfg.MODEL.NET1_CLASSES), dtype=np.float32)
        total_correct_pixels = 0.
        total_valid_pixels = 0.
        angles = []
        for batch_idx, (image, label_1, label_2) in tqdm(enumerate(test_loader)):
            if cfg.CUDA:
                image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()
            out1, out2 = model(image)

            hist, correct_pixels, valid_pixels = compute_hist(out1, label_1, cfg.MODEL.NET1_CLASSES, 255)
            total_hist += hist
            total_correct_pixels += correct_pixels
            total_valid_pixels += valid_pixels

            angle = compute_angle(out2, label_2, 255)
            angles.append(angle)

        IoUs = np.diag(total_hist) / (np.sum(total_hist, axis=0) + np.sum(total_hist, axis=1) - np.diag(total_hist))
        mIoU = np.mean(IoUs)
        print('Mean IoU: {:.3f}'.format(mIoU))

        pixel_acc = total_correct_pixels / total_valid_pixels
        print('Pixel Acc: {:.3f}'.format(pixel_acc))

        angles = np.concatenate(angles, axis=0)
        print('Mean: {:.3f}'.format(np.mean(angles)))
        print('Median: {:.3f}'.format(np.median(angles)))
        print('RMSE: {:.3f}'.format(np.sqrt(np.mean(angles ** 2))))
        print('11.25: {:.3f}'.format(np.mean(np.less_equal(angles, 11.25)) * 100))
        print('22.5: {:.3f}'.format(np.mean(np.less_equal(angles, 22.5)) * 100))
        print('30: {:.3f}'.format(np.mean(np.less_equal(angles, 30.0)) * 100))
        print('45: {:.3f}'.format(np.mean(np.less_equal(angles, 45.0)) * 100))


if __name__ == '__main__':
    main()
