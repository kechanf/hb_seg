import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import cc3d
import os
import matplotlib.pyplot as plt

def save_mip_image(image, filename='', temp_dir="/data/kfchen/nnUNet/temp_mip"):
    if(filename == ''):
        filename = str(np.random.randint(0, 100000)) + '.png'
    mip = np.max(image, axis=0)
    plt.imshow(mip, cmap='gray')
    plt.axis('off')
    plt.title('Maximum Intensity Projection')
    plt.savefig(os.path.join(temp_dir, filename))
    plt.close()
    print(f"MIP图已保存为 {filename}")

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        # print(f"self.data_shape, self.seg_shape {self.data_shape, self.seg_shape} in nnUNetDataLoader3D generate_train_batch")
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            # print(f"data.shape, seg.shape {data.shape, seg.shape} in nnUNetDataLoader3D load_case")
            case_properties.append(properties)

            # print(f"seg.shape: {seg.shape} in generate_train_batch")
            # _, cc_num = cc3d.connected_components(seg[0].copy(), connectivity=26, return_N=True)
            # print(f"cc_num: {cc_num} in generate_train_batch")

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # print(f"padding: {padding}")
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=0)
            seg_all[j] = np.where(seg_all[j] > 0, 1, 0)
            # print(f"data_all[j].shape, seg_all[j].shape {data_all[j].shape, seg_all[j].shape} in nnUNetDataLoader3D")

            # print(f"seg.shape: {seg.shape} in generate_train_batch")
            _, cc_num = cc3d.connected_components(seg_all[j][0].copy(), connectivity=26, return_N=True)
            if(cc_num>1):
                print(f"cc_num: {cc_num} in generate_train_batch")
                save_mip_image(seg_all[j][0])
                data_all[j] = np.zeros_like(data_all[j])
                seg_all[j] = np.zeros_like(seg_all[j])



        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
