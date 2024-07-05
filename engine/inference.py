import logging
import pdb
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from . import eata
from tqdm import tqdm
from utils import comm
from utils.timer import Timer, get_time_str
from collections import defaultdict
from data.datasets.evaluation import evaluate_python
from data.datasets.evaluation import generate_kitti_3d_detection
import ttach as tta
from .visualize_infer import show_image_with_boxes, show_image_with_boxes_test
import torch.nn.functional as F
from structures.image_list import to_image_list


def compute_on_dataset(model, data_loader, device, predict_folder, timer=None, vis=False, 
                        eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    
    model.eval()
    cpu_device = torch.device("cpu")
    dis_ious = defaultdict(list)
    depth_errors = defaultdict(list)

    differ_ious = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
            images = images.to(device)

            # extract label data for visualize
            vis_target = targets[0]
            targets = [target.to(device) for target in targets]

            if timer:
                timer.tic()

            output, eval_utils, visualize_preds = model(images, targets)
            output = output.to(cpu_device)

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            dis_iou = eval_utils['dis_ious']

            if dis_iou is not None:
                for key in dis_iou: dis_ious[key] += dis_iou[key].tolist()

            if vis: show_image_with_boxes(vis_target.get_field('ori_img'), output, vis_target, 
                                    visualize_preds, vis_scores=eval_utils['vis_scores'])

            # generate txt files for predicted objects
            predict_txt = image_ids[0] + '.txt'
            predict_txt = os.path.join(predict_folder, predict_txt)
            generate_kitti_3d_detection(output, predict_txt)

    # disentangling IoU
    for key, value in dis_ious.items():
        mean_iou = sum(value) / len(value)
        dis_ious[key] = mean_iou

    return dis_ious

def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        metrics=['R40'],
        vis=False,
        eval_score_iou=False,
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger("monoflex.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    predict_folder = os.path.join(output_folder, 'data')
    os.makedirs(predict_folder, exist_ok=True)

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    dis_ious = compute_on_dataset(model, data_loader, device, predict_folder, 
                                inference_timer, vis, eval_score_iou)
    comm.synchronize()

    for key, value in dis_ious.items():
        logger.info("{}, MEAN IOU = {:.4f}".format(key, value))

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        return None, None, None

    logger.info('Finishing generating predictions, start evaluating ...')
    ret_dicts = []
    
    for metric in metrics:
        result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
                                        result_path=predict_folder,
                                        label_split_file=dataset.imageset_txt,
                                        current_class=dataset.classes,
                                        metric=metric)

        logger.info('metric = {}'.format(metric))
        logger.info('\n' + result)

        ret_dicts.append(ret_dict)

    return ret_dicts, result, dis_ious

def inference_all_depths(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        vis=False,
        eval_score_iou=False,
        metrics=['R40', 'R11'],
):
    metrics = ['R40']
    inference_timer = None
    device = torch.device(device)
    logger = logging.getLogger("monoflex.inference")
    dataset = data_loader.dataset
    predict_folder = os.path.join(output_folder, 'eval_all_depths')
    os.makedirs(predict_folder, exist_ok=True)
    
    # all methods for solving depths
    class_threshs = [[0.7], [0.5], [0.5]]
    important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
    important_classes = ['Car']

    eval_depth_methods = ['oracle', 'hard', 'soft', 'mean', 'direct', 'keypoints_center', 'keypoints_02', 'keypoints_13']

    eval_depth_dicts = []
    for depth_method in eval_depth_methods:
        logger.info("evaluation with depth method: {}".format(depth_method))
        method_predict_folder = os.path.join(predict_folder, depth_method)
        os.makedirs(method_predict_folder, exist_ok=True)

        # remove all previous predictions
        for file in os.listdir(method_predict_folder):
            os.remove(os.path.join(method_predict_folder, file))

        model.heads.post_processor.output_depth = depth_method
        dis_ious = compute_on_dataset(model, data_loader, device, method_predict_folder, 
                                    inference_timer, vis, eval_score_iou)
        result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
                                        result_path=method_predict_folder,
                                        label_split_file=dataset.imageset_txt,
                                        current_class=dataset.classes,
                                        metric='R40')
        
        eval_depth_dicts.append(ret_dict)

    for cls_idx, cls in enumerate(important_classes):
        cls_thresh = class_threshs[cls_idx]
        for thresh in cls_thresh:
            logger.info('{} AP@{:.2f}, {:.2f}:'.format(cls, thresh, thresh))
            sort_metric = []
            for depth_method, eval_depth_dict in zip(eval_depth_methods, eval_depth_dicts):
                logger.info('bev/3d AP, method {}:'.format(depth_method))
                logger.info('{:.4f}/{:.4f}, {:.4f}/{:.4f}, {:.4f}/{:.4f}'.format(eval_depth_dict['{}_bev_{:.2f}/easy'.format(cls, thresh)], 
                        eval_depth_dict['{}_3d_{:.2f}/easy'.format(cls, thresh)], eval_depth_dict['{}_bev_{:.2f}/moderate'.format(cls, thresh)], 
                        eval_depth_dict['{}_3d_{:.2f}/moderate'.format(cls, thresh)], eval_depth_dict['{}_bev_{:.2f}/hard'.format(cls, thresh)],
                        eval_depth_dict['{}_3d_{:.2f}/hard'.format(cls, thresh)]))

                sort_metric.append(eval_depth_dict['{}_3d_{:.2f}/moderate'.format(cls, thresh)])

            sort_metric = np.array(sort_metric)
            sort_idxs = np.argsort(-sort_metric)
            join_str = ' > '
            sort_str = join_str.join([eval_depth_methods[idx] for idx in sort_idxs])
            logger.info('Cls {}, Thresh {}, Sort: '.format(cls, thresh) + sort_str)

    return None, None, None

def tta_monoflex_ours(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        metrics=['R40'],
        vis=False,
        eval_score_iou=False,
        clean_model=None
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger("monoflex.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    predict_folder = os.path.join(output_folder, 'data')
    os.makedirs(predict_folder, exist_ok=True)

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    tta_model = tta_by_score(model, data_loader, device, predict_folder, inference_timer, vis, eval_score_iou)
    comm.synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        return None, None, None

    logger.info('Finishing generating predictions, start evaluating ...')
    
    ret_dicts = []
    for metric in metrics:
        result, ret_dict = evaluate_python(label_path=dataset.label_dir, 
                                        result_path=predict_folder,
                                        label_split_file=dataset.imageset_txt,
                                        current_class=dataset.classes,
                                        metric=metric)

        logger.info('metric = {}'.format(metric))
        logger.info('\n' + result)

    ret_dicts.append(ret_dict)

    return tta_model

def tta_by_score(model, data_loader, device, predict_folder, timer=None, vis=False,
                 eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    lr = 0.5 * 1e-3
    model = eata.configure_model(model)
    params, param_names = eata.collect_params(model)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
        ]
    )
    cls_nums = 3
    temprature = 1.0
    cpu_device = torch.device("cpu")
    for idx, batch in enumerate(tqdm(data_loader)):
        model.train()
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        if timer:
            timer.tic()

        # simple test-time augmentations
        aug_scores = []
        aug_total_scores = []
        aug_probs = []

        for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform()
            tmp_imgs = images.tensors
            aug_imgs = transformer.augment_image(tmp_imgs)
            scores, score_dict, prob_dict = model(aug_imgs, targets, tta=True, temp=1.0)

            aug_scores.append(scores)
            aug_total_scores.append(score_dict)
            aug_probs.append(prob_dict)

        # high-confidence detection optimization
        debug_mean_score = torch.tensor(0.).cuda()
        tmp_len = 0
        for each_score in aug_scores:
            debug_mask = each_score >= 0.2  # remove the background part, following the orignal setting
            if debug_mask.sum() > 0:
                debug_mean_score += torch.mean(each_score[debug_mask])
                tmp_len += len(each_score[debug_mask])
        debug_mean_score /= len(aug_scores)
        tmp_len /= len(aug_scores)
        
        if idx == 0:
            cur_threshold = 0.2
            pre_threshold = 0.2
            alpha = 0.1
        elif tmp_len > 0:
            cur_threshold = alpha * debug_mean_score + (1 - alpha) * pre_threshold
            pre_threshold = cur_threshold
        print('iteration {}, cur_threshold {}, pre_threshold {}'.format(idx, cur_threshold, pre_threshold))
        confs_loss = torch.tensor(0.).cuda()
        NL_loss = torch.tensor(0.).cuda()
        bs = images.tensors.shape[0]
        for all_score_view in aug_total_scores:
            batch_loss = torch.tensor(0.).cuda()
            cnt_obj = 0

            cls_NL_loss = torch.zeros(cls_nums).cuda()
            cls_sample_num = [0] * cls_nums
            for i in range(bs):
                cur_preds = all_score_view[i]
                for each_pred in cur_preds:
                    max_idx = torch.argmax(each_pred)
                    # ori monotta
                    if each_pred[max_idx] > cur_threshold:
                        batch_loss = batch_loss + (-1 * torch.log(each_pred[max_idx]))
                        cnt_obj += 1

                    elif each_pred[max_idx] > 0.05:  # too dig out more potential objects(affected by the noise)
                        # negative part: Rebalance negative loss
                        all_idx = [0, 1, 2]  # car, XX, XX --> class-level
                        all_idx.remove(max_idx)
                        neg_idx = np.random.choice(all_idx)
                        neg_pred = F.softmax(each_pred / temprature)
                        tmp_nl_loss = -(1 - neg_pred[neg_idx].detach()) * torch.log(1 - neg_pred[neg_idx])
                        cls_NL_loss[neg_idx] += tmp_nl_loss
                        cls_sample_num[neg_idx] += 1                

            print('Detect {} positive objects, {} potential objects'.format(cnt_obj, sum(cls_sample_num)))
            for i in range(cls_nums):
                if cls_sample_num[i] > 0:
                    cls_NL_loss[i] = cls_NL_loss[i] / cls_sample_num[i]

            confs_loss = confs_loss + batch_loss / cnt_obj
            NL_loss = NL_loss + torch.sum(cls_NL_loss)

        confs_loss = confs_loss / len(aug_total_scores)
        NL_loss = NL_loss / len(aug_total_scores)

        uncertainty_loss = confs_loss + 1 * NL_loss
        uncertainty_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                
        print(np.round(confs_loss.item(), 3), np.round(NL_loss.item(), 3), "score, NL")
        print(tmp_len, np.round(debug_mean_score.item(), 3))

        if timer:
            torch.cuda.synchronize()
            timer.toc()
        
        # start eval
        with torch.no_grad():
            model.eval()
            eval_images = to_image_list(images)
            for i in range(bs):
                output, eval_utils, visualize_preds = model(eval_images.tensors[i].unsqueeze(0), [targets[i]], tta_eval=True)
                output = output.to(cpu_device)             
                # generate txt files for predicted objects
                predict_txt = image_ids[i] + '.txt'

                if not os.path.exists(predict_folder):
                    os.makedirs(predict_folder)

                predict_txt = os.path.join(predict_folder, predict_txt)
                generate_kitti_3d_detection(output, predict_txt)
    return model