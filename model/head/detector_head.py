import torch
from torch import nn
import pdb

from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor

class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = make_predictor(cfg, in_channels)
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)

    def forward(self, features, targets=None, test=False, tta=False, temp=None, tta_eval=False, detected_threshold=None, update_return=False):
        x = self.predictor(features, targets)

        if tta and not update_return:
            result, eval_utils, visualize_preds = self.post_processor.tta_forward_1016(x, targets, test=test, features=features, temp=temp)
            return result, eval_utils, visualize_preds
        elif tta and update_return:
            result, eval_utils, tta_outputs = self.post_processor.tta_forward_240525_update(x, targets, test=test, features=features, temp=temp, detected_threshold=detected_threshold)
            return result, eval_utils, tta_outputs
        result, eval_utils, visualize_preds = self.post_processor(x, targets, test=test, features=features)
        return result, eval_utils, visualize_preds
    
def bulid_head(cfg, in_channels):
    
    return Detect_Head(cfg, in_channels)