import argparse
import copy
import glob
from pathlib import Path
import random
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, NuScenesDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.datasets import __all__ as ALL_DATASETS

from pcdet.utils import common_utils
from pcdet.utils import transform_utils

# from pcdet.models.dense_heads.coco_classes import coco_classes
import json
import tqdm

import os
from pathlib import Path

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/uqdetche/OpenPCDet/tools/cfgs/nuscenes_models/frustum_proposals.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--folder', type=str, default=None, help='folder name for extracted pseudo labels')
    parser.add_argument('--workers', type=int, default=1, help='workers')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')


    args = parser.parse_args()
    assert args.batch_size == 1, 'have not implemented pred saving for bs > 1'

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    dataset_cfg = cfg.DATA_CONFIG
    INF_CONFIG = copy.deepcopy(cfg.DATA_CONFIG)
    print('augmentor', INF_CONFIG['DATA_AUGMENTOR'])
    print('aug list', INF_CONFIG['DATA_AUGMENTOR']['AUG_CONFIG_LIST'])
    INF_CONFIG['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] = []
    inf_set, inf_loader, inf_sampler = build_dataloader(
        dataset_cfg=INF_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True, #?
        merge_all_iters_to_one_epoch=False,
        total_epochs=1,
        # seed=666
    )
    
    # if inf_set.training:
        # print('dataset len', len(inf_set), inf_set._merge_all_iters_to_one_epoch)
        # assert len(inf_set) == 28130
    # exit()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inf_set)

    model.eval()

    if args.folder is None:
        labels_id = model.dense_head._get_name() + '_owlvit'
        base_path = f"../data/pseudo_labels/{labels_id}/"

    else:
        labels_id = args.folder
        base_path = labels_id

    print('labels_id', labels_id)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    else:
        print('folder exists, quitting', base_path)
        exit()

    # idx_to_process = []
    # num_to_process = 0
    # # see if any do not exist
    # with tqdm(total=len(demo_dataset), desc='Extracting Pseudo-labels') as pbar:
    #     for idx, data_dict in enumerate(demo_dataset):
    #         # data_dict = demo_dataset.collate_batch([data_dict])

    #         # frame_ids = data_dict['frame_id']
    #         # assert len(frame_ids) == 1, frame_ids

    #         frame_id = data_dict['frame_id']

    #         pseudo_path = f"{base_path}/{frame_id.replace('.', '_')}.pth"

    #         if not os.path.exists(pseudo_path):
    #             num_to_process += 1
    #             idx_to_process.append(idx)

    #         pbar.update(1)
    #         pbar.set_postfix(ordered_dict=dict(num_to_process=num_to_process, frame_id=frame_id))


    thresh_list = [0.3, 0.5, 0.7]
    curr_sum = [0] * len(thresh_list)
    curr_n = 0
    num_failed = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(inf_loader), leave=True, desc='Extracting Pseudolabels', dynamic_ncols=True)
        for i, data_dict in enumerate(inf_loader):
            load_data_to_gpu(data_dict)
            frame_ids = data_dict['frame_id']

            # print('i', i, 'frameid', frame_ids)

            data_dict['batch_idx'] = i
            pred_dicts, _ = model.forward(data_dict)

            recall_dict = Detector3DTemplate.generate_recall_record(pred_dicts[0]['pred_boxes'], dict(), 0, data_dict, thresh_list=thresh_list)

            log_dict = {}
            curr_n += recall_dict['gt']

            for i, thr in enumerate(thresh_list):
                curr_sum[i] += recall_dict[f'rcnn_{thr:.1f}']
                log_dict[f'rcnn_{thr:.1f}'] = curr_sum[i] / curr_n

            curr_failed = []
            for frame_id in frame_ids:
                save_path = Path(base_path) / f"{frame_id.replace('.', '_')}.pth"

                torch.save(pred_dicts, save_path)

                if not os.path.exists(save_path):
                    num_failed += 1
                    print('failed', pred_dicts)

            log_dict['num_faileded'] = num_failed

            pbar.update(1)
            pbar.set_postfix(ordered_dict=log_dict)

    logger.info('extraction done.')


if __name__ == '__main__':
    main()
