from nnunetv2.training.loss.fmm_preprocess2 import find_path_to_source
import numpy as np
import cc3d

def npathloss(gt, pred, predecessor, num_paths=10):
    gt_clone = gt.detach().clone()
    pred_clone = pred.detach().clone()
    predecessor_clone = predecessor.detach().clone()
    
    connected_components = cc3d.connected_components(predecessor_clone.cpu().numpy())
    
