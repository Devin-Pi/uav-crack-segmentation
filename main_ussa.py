import argparse
import comet_ml
import torch
from torch.utils.data import DataLoader

import lightning as l
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.loggers import CometLogger

from model.ussa import UNet
from utils.load_config import load_config
from dataload.dataload import make_datapath_list, VOCDataset, DataTransform




def main(args):

    train_config = load_config(args.train_config_path)

    l.seed_everything(1744)

    model = UNet(c_in = 3, c_out = 2 )

    data_path = train_config['data_path']
    train_img_list, train_anno_list, val_img_list, val_anno_list, _, _ = make_datapath_list(data_path)

    color_mean = (0.485, 0.456, 0.406)
    clolor_std = (0.229, 0.224, 0.225)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                               n_classes=train_config['n_classes'], input_shape=train_config['input_shape'],
                               transform=DataTransform(input_size=256, color_mean=color_mean, color_std=clolor_std))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                             n_classes=train_config['n_classes'], input_shape=train_config['input_shape'],
                             transform=DataTransform(input_size=256, color_mean=color_mean, color_std=clolor_std))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  num_workers=train_config['num_workers'],
                                #   shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=train_config['batch_size'],
                                  num_workers=train_config['num_workers'],
                                #   shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)

    model_checpoint = ModelCheckpoint(dirpath='/unet-semantic/check_point',
                                      filename="{epoch}-{val_loss:.2f}-{train_miou:.2f}",
                                      save_last=True,
                                      every_n_epochs=20,
                                      save_top_k=-1)

    comet_logger = CometLogger(api_key='w0Npr1BzeXnixD0UOBF6onh5d',
                               project_name='crack_segmentation',
                               experiment_name='deepcrack500_2sp_1att',
                               workspace = 'semantic',
                               save_dir='.',
                               )

    trainer = l.Trainer(
        # strategy='ddp_find_unused_parameters_true',
        strategy="ddp",
        # precision='16-mixed',
        max_epochs=300,
        sync_batchnorm=True,
        # fast_dev_run=10,
        callbacks=[model_checpoint],
        logger=comet_logger,
    )

    trainer.fit(model = model,
                train_dataloaders = train_dataloader,
                val_dataloaders = val_dataloader,
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_config_path",
                        default='/unet-semantic/config/train_config.yaml',
                        type=str,
                        help='the path for train config yaml file')

    args = parser.parse_args()

    main(args)
