import os, glob
import os.path as osp
import numpy as np
import imageio



def main():
    src_long_dir = '/data/gjh8760/Dataset/sid_processed/long_sid2'
    src_short_dir = '/data/gjh8760/Dataset/sid_processed/short_sid2'
    dst_long_dir = '/data/gjh8760/Dataset/sid_processed_patch/long_sid2'
    dst_short_dir = '/data/gjh8760/Dataset/sid_processed_patch/short_sid2'
    stride_h = 128
    stride_w = 176
    patch_size = 256

    os.makedirs(dst_long_dir, exist_ok=True)
    os.makedirs(dst_short_dir, exist_ok=True)

    subfolder_list = sorted(os.listdir(src_long_dir))
    for subfolder_name in subfolder_list:
        # GT
        subfolder_paths_GT = sorted(glob.glob(osp.join(src_long_dir, subfolder_name, '*.npy')))
        for subfolder_path_GT in subfolder_paths_GT:
            GT_img = np.load(subfolder_path_GT)
            name_GT = osp.basename(subfolder_path_GT)
            h, w, c = GT_img.shape
            for i in range(0, h, stride_h):
                for j in range(0, w, stride_w):
                    GT_patch = GT_img[i:i + patch_size, j:j + patch_size, :]
                    name_GT_patch = name_GT[:-4] + f'_{str(i)}_{str(j)}.npy'
                    os.makedirs(f'{dst_long_dir}/{subfolder_name}/{str(i)}_{str(j)}', exist_ok=True)
                    np.save(f'{dst_long_dir}/{subfolder_name}/{str(i)}_{str(j)}/{name_GT_patch}', GT_patch)
        
        # LQ
        subfolder_paths_LQ = sorted(glob.glob(osp.join(src_short_dir, subfolder_name, '*.npy')))
        for subfolder_path_LQ in subfolder_paths_LQ:
            LQ_img = np.load(subfolder_path_LQ)
            name_LQ = osp.basename(subfolder_path_LQ)
            h, w, c = LQ_img.shape
            for i in range(0, h, stride_h):
                for j in range(0, w, stride_w):
                    LQ_patch = LQ_img[i:i + patch_size, j:j + patch_size, :]
                    name_LQ_patch = name_LQ[:-4] + f'_{str(i)}_{str(j)}.npy'
                    os.makedirs(f'{dst_short_dir}/{subfolder_name}/{str(i)}_{str(j)}', exist_ok=True)
                    np.save(f'{dst_short_dir}/{subfolder_name}/{str(i)}_{str(j)}/{name_LQ_patch}', LQ_patch)



if __name__ == '__main__':
    main()