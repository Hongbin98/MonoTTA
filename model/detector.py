import torch
from torch import nn

from structures.image_list import to_image_list

from .backbone import build_backbone
from .head.detector_head import bulid_head

from model.layers.uncert_wrapper import make_multitask_wrapper

class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = bulid_head(cfg, self.backbone.out_channels)
        self.test = cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, targets=None, tta=False, temp=None, tta_eval=False, return_feas=False, detected_threshold=None, update_return=False):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        if not tta_eval:
            images = to_image_list(images)
            features = self.backbone(images.tensors)
        else:
            features = self.backbone(images)
            result, eval_utils, visualize_preds = self.heads(features[-1], targets, test=True, tta_eval=True, detected_threshold=detected_threshold)
            return result, eval_utils, visualize_preds

        if tta and not update_return:
            if not return_feas:
                result, cls_preds, visualize_preds = self.heads(features[-1], targets, test=self.test, tta=True, temp=temp)
                return result, cls_preds, visualize_preds
            else:
                result, cls_preds, selected_feas = self.heads(features[-1], targets, test=self.test, tta=True, temp=temp, update_return=update_return)
                return result, selected_feas, features                
        
        if tta and update_return:
                result, cls_preds, tta_outputs = self.heads(features[-1], targets, test=self.test, tta=True, temp=temp, update_return=update_return, detected_threshold=detected_threshold)
                tta_outputs['bab_feas'] = features
                return result, cls_preds, tta_outputs            

        if self.training:
            loss_dict, log_loss_dict = self.heads(features[-1], targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.heads(features[-1], targets, test=self.test)
            return result, eval_utils, visualize_preds
