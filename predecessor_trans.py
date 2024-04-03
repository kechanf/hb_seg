from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from nnunetv2.training.loss.fmm_preprocess2 import get_fmm_from_img
import time
import os


class AddPredecessorImageTransform(AbstractTransform):
    def __init__(self, predecessor_key: str = "predecessor", target_key: str = "target"):
        """
        Adds the predecessor image to the data dictionary. This transform assumes that the predecessor
        image is already computed and available in the data_dict under the key specified by predecessor_key.
        """
        self.predecessor_key = predecessor_key
        self.target_key = target_key

    def __call__(self, **data_dict):
        target_shape = data_dict[self.target_key].shape
        data_dict[self.predecessor_key] = np.zeros(target_shape, dtype=np.int32) # (2, 1, 48, 224, 224)
        target = data_dict[self.target_key]
        for batch_idx in range(target.shape[0]):
            for channel_idx in range(target.shape[1]):
                current_target = target[batch_idx, channel_idx]
                current_predecessor = get_fmm_from_img(current_target)
                data_dict[self.predecessor_key][batch_idx, channel_idx] = current_predecessor

        return data_dict
