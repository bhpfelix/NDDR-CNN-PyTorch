import os
import argparse
from configs import cfg

import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import MultiTaskDataset

from models.nddr_net import NDDRNet
from models.vgg16_lfov import DeepLabLargeFOV

from utils.losses import get_normal_loss

from eval import evaluate

import datetime
from tensorboardX import SummaryWriter
from utils.visualization import process_image, process_seg_label, process_normal_label


def main():
    parser = argparse.ArgumentParser(description="PyTorch NDDR Training")
    parser.add_argument(
        "--config-file",
        default="vgg16_nddr_pret.yaml",
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

    if not os.path.exists(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME)):
        os.makedirs(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME))

    # load the data
    train_loader = torch.utils.data.DataLoader(
        MultiTaskDataset(
            data_dir=cfg.DATA_DIR,
            data_list_1=cfg.TRAIN.DATA_LIST_1,
            data_list_2=cfg.TRAIN.DATA_LIST_2,
            output_size=cfg.TRAIN.OUTPUT_SIZE,
            random_scale=cfg.TRAIN.RANDOM_SCALE,
            random_mirror=cfg.TRAIN.RANDOM_MIRROR,
            random_crop=cfg.TRAIN.RANDOM_CROP,
            ignore_label=cfg.IGNORE_LABEL,
        ),
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    # load the data
    if cfg.TRAIN.EVAL_CKPT:
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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
    experiment_log_dir = os.path.join(cfg.LOG_DIR, cfg.EXPERIMENT_NAME, timestamp)
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)
    writer = SummaryWriter(logdir=experiment_log_dir)

    net1 = DeepLabLargeFOV(3, cfg.MODEL.NET1_CLASSES, weights=cfg.TRAIN.WEIGHT_1)
    net2 = DeepLabLargeFOV(3, cfg.MODEL.NET2_CLASSES, weights=cfg.TRAIN.WEIGHT_2)
    model = NDDRNet(net1, net2,
                    shortcut=cfg.MODEL.SHORTCUT,
                    bn_before_relu=cfg.MODEL.BN_BEFORE_RELU)
    
    if cfg.CUDA:
        model = model.cuda()
    model.train()
    steps = 0

    seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    # hacky way to pick params
    nddr_params = []
    fc8_weights = []
    fc8_bias = []
    base_params = []
    for k, v in model.named_parameters():
        if 'nddrs' in k:
            nddr_params.append(v)
        elif 'head.7' in k:
            if 'weight' in k:
                fc8_weights.append(v)
            else:
                assert 'bias' in k
                fc8_bias.append(v)
        else:
            base_params.append(v)
    assert len(nddr_params) > 0 and len(fc8_weights) > 0 and len(fc8_bias) > 0

    parameter_dict = [
        {'params': base_params},
        {'params': fc8_weights, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_WEIGHT_FACTOR},
        {'params': fc8_bias, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_BIAS_FACTOR},
        {'params': nddr_params, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NDDR_FACTOR}
    ]
    optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda step: (1 - float(step) / cfg.TRAIN.STEPS)**cfg.TRAIN.POWER,
                                            last_epoch=-1)

    while steps < cfg.TRAIN.STEPS:
        for batch_idx, (image, label_1, label_2) in enumerate(train_loader):
            if cfg.CUDA:
                image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()
            optimizer.zero_grad()
            out1, out2 = model(image)

            # loss_seg = get_seg_loss(out1, label_1, 40, 255)
            loss_seg = seg_loss(out1, label_1.squeeze(1))
            loss_normal = get_normal_loss(out2, label_2, 255)

            loss = loss_seg + cfg.TRAIN.NORMAL_FACTOR * loss_normal

            loss.backward()

            optimizer.step()
            scheduler.step()

            # Print out the loss periodically.
            if steps % cfg.TRAIN.LOG_INTERVAL == 0:
                print('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                    steps, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(),
                    loss_seg.data.item(), loss_normal.data.item()))

                # Log to tensorboard
                writer.add_scalar('lr', scheduler.get_lr()[0], steps)
                writer.add_scalar('loss/overall', loss.data.item(), steps)
                writer.add_scalar('loss/seg', loss_seg.data.item(), steps)
                writer.add_scalar('loss/normal', loss_normal.data.item(), steps)

                writer.add_image('image', process_image(image[0]), steps)
                seg_pred, seg_gt = process_seg_label(
                    out1.argmax(dim=1)[0].detach(),
                    label_1.squeeze(1)[0],
                    cfg.MODEL.NET1_CLASSES
                )
                writer.add_image('seg/pred', seg_pred, steps)
                writer.add_image('seg/gt', seg_gt, steps)
                normal_pred, normal_gt = process_normal_label(out2[0].detach(), label_2[0], 255)
                writer.add_image('normal/pred', normal_pred, steps)
                writer.add_image('normal/gt', normal_gt, steps)

            if steps % cfg.TRAIN.SAVE_INTERVAL == 0:
                checkpoint = {
                    'cfg': cfg,
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'loss_seg': loss_seg,
                    'loss_normal': loss_normal,
                    'mIoU': None,
                    'PixelAcc': None,
                    'angle_metrics': None,
                }

                if cfg.TRAIN.EVAL_CKPT:
                    model.eval()
                    mIoU, pixel_acc, angle_metrics = evaluate(test_loader, model)
                    writer.add_scalar('eval/mIoU', mIoU, steps)
                    writer.add_scalar('eval/PixelAcc', pixel_acc, steps)
                    for k, v in angle_metrics:
                        writer.add_scalar('eval/{}'.format(k), v, steps)
                    checkpoint['mIoU'] = mIoU
                    checkpoint['PixelAcc'] = pixel_acc
                    checkpoint['angle_metrics'] = angle_metrics
                    model.train()

                torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME,
                                                    'ckpt-%s.pth' % str(steps).zfill(5)))

            steps += 1
            if steps >= cfg.TRAIN.STEPS:
                break


if __name__ == '__main__':
    main()
