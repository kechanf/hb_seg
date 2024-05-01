from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from nnunetv2.training.loss.fmm.fmm_process import get_fmm_from_img
import time
import os
import cc3d


class AddPredecessorImageTransform(AbstractTransform):
    def __init__(self, predecessor_key: str = "predecessor", target_key: str = "target", soma_key: str = "soma"):
        """
        Adds the predecessor image to the data dictionary. This transform assumes that the predecessor
        image is already computed and available in the data_dict under the key specified by predecessor_key.
        """
        self.predecessor_key = predecessor_key
        self.target_key = target_key
        self.soma_key = soma_key

    def __call__(self, **data_dict):
        target_shape = data_dict[self.target_key].shape
        data_dict[self.predecessor_key] = np.zeros(target_shape, dtype=np.int32) # (2, 1, 48, 224, 224)
        data_dict[self.soma_key] = np.zeros(tuple([target_shape[0], target_shape[1], 3]), dtype=np.int32)
        target = data_dict[self.target_key]

        # print(target[0].shape, target.dtype, len(target))

        for batch_idx in range(target.shape[0]):
            for channel_idx in range(target.shape[1]):
                current_target = target[batch_idx, channel_idx]
                current_predecessor, current_soma = get_fmm_from_img(current_target)
                if((current_predecessor is None) or (current_soma is None)):
                    # print("... no predecessor found for batch_idx: ", batch_idx, " channel_idx: ", channel_idx)
                    data_dict[self.predecessor_key][batch_idx, channel_idx] = np.ones_like(current_target) * -1
                    data_dict[self.soma_key][batch_idx, channel_idx] = np.zeros(3)
                else:
                    data_dict[self.predecessor_key][batch_idx, channel_idx] = current_predecessor
                    data_dict[self.soma_key][batch_idx, channel_idx] = current_soma

        return data_dict