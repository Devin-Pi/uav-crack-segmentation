# how to conduct evaluation between label and segmentation results?
import torch
from torch.utils.data import DataLoader
import lightning as l
from PIL import Image
import numpy as np
# from model.unet import UNet
from model.ussa import UNet
from utils.load_config import load_config
from dataload.dataload import make_datapath_list, VOCDataset, DataTransform
from torchmetrics import JaccardIndex
import hydra
from omegaconf import DictConfig, OmegaConf
import mmcv
# from mmseg.evaluation.metrics.iou_metric import IoUMetric
from mmseg.core.evaluation.metrics import mean_iou, mean_dice

from utils.img_save import save_mask
from model.pspnet_lightning import PSPNet
from model.segnet import SegNet


@hydra.main(version_base=None, config_path='/unet-semantic/config', config_name='train_config')
def main(cfg: DictConfig) -> None:
    # train_config = load_config(cfg.train_config_path)

    l.seed_everything(1744)

    model = UNet(c_in=3, c_out=2)
    # model = PSPNet(n_classes=2)
    # model = SegNet(cfg)

    model = model.load_from_checkpoint(
        # "check_point/unet_origin_.ckpt",
        # "check_point/unet_2spb_1att.ckpt",
        "/unet-semantic/check_point/last.ckpt",
        # "check_point/ablation_1_spb128.ckpt",
                                   map_location='cpu')
    data_path = cfg.data_path #train_config['data_path']
    _, _, _, _, test_img_list, test_anno_list = make_datapath_list(data_path)

    color_mean = (0.485, 0.456, 0.406)
    clolor_std = (0.229, 0.224, 0.225)

    test_dataset = VOCDataset(test_img_list, test_anno_list, phase='test',
                               n_classes=cfg.n_classes, #train_config['n_classes'],
                               input_shape=cfg.input_shape, #  train_config['input_shape'],
                               transform=DataTransform(input_size=256,
                                                       color_mean=color_mean,
                                                       color_std=clolor_std))

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1, # train_config['batch_size'],
                                  num_workers=cfg.num_workers, # train_config['num_workers'],
                                #   shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    all_miou = []
    all_dice = []
    model.eval()
    for n, batch in enumerate(test_dataloader):
        img, mask = batch
        # img: [bs, 3, h, w]
        # msk: [bs, h, w]
        # pred: [bs, n_classes, h, w]
        # with torch.no_grad():

        pred = model(img)
        pred = pred[0].detach().numpy()
        pred = np.argmax(pred, axis = 0)

        # save_mask(img=test_img_list[n],
        #           pred=pred)
        # save_mask(img_list=test_img_list[n],
        #             result=pred,
        #             n_classes=[2],
        #             palette=None,
        #             out_file="/unet-semantic/result/",
        #             idx = n)

        mask = mask[0].numpy()
        iou = mean_iou(pred, mask, 2, 0) # the num_class & ignore_index ???
        A = iou["IoU"][1]
        dice = mean_dice(pred, mask, 2, 0)
        b = dice["Dice"][1]
        miou = round(np.nanmean(iou["IoU"][1]) * 100, 3)
        dice = round(np.nanmean(dice["Dice"][1]) * 100, 2)
        all_miou.append(miou)
        all_dice.append(dice)
        with open(cfg.result_dir + 'miou.txt', 'a') as f:
            print(f'{n} {miou}', file =f)
        with open(cfg.result_dir + 'dice.txt', 'a') as f:
            print(f'{n} {dice}', file =f)
    print(f'mean_iou = {np.mean(all_miou)}')
    print(f'mean_dice = {np.mean(all_dice)}')

if __name__ == "__main__":
    main()