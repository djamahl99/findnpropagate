import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def statistics_info(cfg, ret_dict, metric, disp_dict):
    metric['num_3known'] += ret_dict.get('num_3known', 0)
    metric['num_6known'] += ret_dict.get('num_6known', 0)
    metric['num_7unknown'] += ret_dict.get('num_7unknown', 0)
    metric['num_4unknown'] += ret_dict.get('num_4unknown', 0)

    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)

        metric['rcnn_3known_%s' % (str(cur_thresh))] += ret_dict.get('rcnn_3known_%s' % (str(cur_thresh)), 0)
        metric['rcnn_6known_%s' % (str(cur_thresh))] += ret_dict.get('rcnn_6known_%s' % (str(cur_thresh)), 0)
        metric['rcnn_7unknown_%s' % (str(cur_thresh))] += ret_dict.get('rcnn_7unknown_%s' % (str(cur_thresh)), 0)
        metric['rcnn_4unknown_%s' % (str(cur_thresh))] += ret_dict.get('rcnn_4unknown_%s' % (str(cur_thresh)), 0)

    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

    disp_dict['rcnn_3known_%s' % (str(min_thresh))] = '%d / %d' % (metric.get('rcnn_3known_%s' % (str(min_thresh)), 0), max(metric['num_3known'], 1))
    disp_dict['rcnn_6known_%s' % (str(min_thresh))] = '%d / %d' % (metric.get('rcnn_6known_%s' % (str(min_thresh)), 0), max(metric['num_6known'], 1))
    disp_dict['rcnn_7unknown_%s' % (str(min_thresh))] = '%d / %d' % (metric.get('rcnn_7unknown_%s' % (str(min_thresh)), 0), max(metric['num_7unknown'], 1))
    disp_dict['rcnn_4unknown_%s' % (str(min_thresh))] = '%d / %d' % (metric.get('rcnn_4unknown_%s' % (str(min_thresh)), 0), max(metric['num_4unknown'], 1))

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
        'num_3known': 0,
        'num_6known': 0,
        'num_4unknown': 0,
        'num_7unknown': 0
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        metric['rcnn_3known_%s' % (str(cur_thresh))] = 0
        metric['rcnn_6known_%s' % (str(cur_thresh))] = 0

        # unknowns
        metric['rcnn_4unknown_%s' % (str(cur_thresh))] = 0
        metric['rcnn_7unknown_%s' % (str(cur_thresh))] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    # class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
    #           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    # print("dataset.class_names", dataset.class_names, len(dataset.class_names))
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        # num_3known = ret_dict['num_3known']
        # num_6known = ret_dict['num_6known']
        # num_7unknown = ret_dict['num_7unknown']
        # num_4unknown = ret_dict['num_4unknown']

        # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        #     ret_dict['rcnn_3known_%s' % (str(cur_thresh))] = ret_dict['rcnn_3known_%s' % (str(cur_thresh))] / max(num_3known, 1)
        #     ret_dict['rcnn_6known_%s' % (str(cur_thresh))] = ret_dict['rcnn_6known_%s' % (str(cur_thresh))] / max(num_6known, 1)
        #     ret_dict['rcnn_7unknown_%s' % (str(cur_thresh))] = ret_dict['rcnn_7unknown_%s' % (str(cur_thresh))] / max(num_7unknown, 1)
        #     ret_dict['rcnn_4unknown_%s' % (str(cur_thresh))] = ret_dict['rcnn_4unknown_%s' % (str(cur_thresh))] / max(num_4unknown, 1)

        #     logger.info('rcnn_3known_%s: %f' % (cur_thresh, ret_dict['rcnn_3known_%s' % (str(cur_thresh))]))
        #     logger.info('rcnn_6known_%s: %f' % (cur_thresh, ret_dict['rcnn_6known_%s' % (str(cur_thresh))]))
        #     logger.info('rcnn_7unknown_%s: %f' % (cur_thresh, ret_dict['rcnn_7unknown_%s' % (str(cur_thresh))]))
        #     logger.info('rcnn_4unknown_%s: %f' % (cur_thresh, ret_dict['rcnn_4unknown_%s' % (str(cur_thresh))]))

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    num_3known = metric['num_3known']
    num_6known = metric['num_6known']
    num_7unknown = metric['num_7unknown']
    num_4unknown = metric['num_4unknown']

    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall
        
        ret_dict['rcnn_3known_%s' % (str(cur_thresh))] = metric['rcnn_3known_%s' % (str(cur_thresh))] / max(num_3known, 1)
        ret_dict['rcnn_6known_%s' % (str(cur_thresh))] = metric['rcnn_6known_%s' % (str(cur_thresh))] / max(num_6known, 1)
        ret_dict['rcnn_7unknown_%s' % (str(cur_thresh))] = metric['rcnn_7unknown_%s' % (str(cur_thresh))] / max(num_7unknown, 1)
        ret_dict['rcnn_4unknown_%s' % (str(cur_thresh))] = metric['rcnn_4unknown_%s' % (str(cur_thresh))] / max(num_4unknown, 1)

        logger.info('rcnn_3known_%s: %f' % (cur_thresh, ret_dict['rcnn_3known_%s' % (str(cur_thresh))]))
        logger.info('rcnn_6known_%s: %f' % (cur_thresh, ret_dict['rcnn_6known_%s' % (str(cur_thresh))]))
        logger.info('rcnn_7unknown_%s: %f' % (cur_thresh, ret_dict['rcnn_7unknown_%s' % (str(cur_thresh))]))
        logger.info('rcnn_4unknown_%s: %f' % (cur_thresh, ret_dict['rcnn_4unknown_%s' % (str(cur_thresh))]))
        
            # ret_dict[f'recall/{k}{str(cur_thresh)}'] = 

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # print('ret_dict')
    for k, v in ret_dict.items():
        logger.info(f'{k} -> {v}')


    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
