import os
from decord import VideoReader
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision.transforms import Compose, Lambda
import h5py
import re

#this part will creat mask and the dataset

transform = Compose([
    Lambda(lambda x: torch.tensor(x))  # Convert to tensor
])

def create_mask(video_tensor, mask_type='sti', mask_file=None, block_sizes=[4], keep=4, interval=[2,5]):
   
    """
    Create a mask for the video tensor with five types: 'sti', 'fi', 'nowcasting', 'stin', or 'stis'.

    Args:
        video_tensor (torch.Tensor): The video tensor with shape (T, H, W, C).
        mask_type (str): Type of the mask to create. 'sti' for spatio-temporal interpolation mask,
                        'fi' for frame interpolation mask, 'nowcasting' for future rainfall prediction mask,
                        'stin' for spatio-temporal simulation mask, 'stis' for custom spatio-temporal interpolation mask.

    Returns:
        torch.Tensor: A boolean mask tensor with the same shape as video_tensor,
        where True indicates the element is masked.
    """

    T, H, W, C = video_tensor.shape
    mask = torch.zeros((T, H, W, C), dtype=torch.float32)
   
    if mask_type == 'sti':
        # 創建一個 3D mask_matrix (1, H, W, C)
        mask_matrix = torch.zeros((1, H, W, C), dtype=torch.float32)
        
        # 隨機選擇一個塊大小
        block_size = np.random.choice(block_sizes)
        h_start = 0
        while h_start < H:
            w_start = 0
            while w_start < W:
                # 確保不會超出圖像邊界
                h_end = min(h_start + block_size, H)
                w_end = min(w_start + block_size, W)
                
                # 在塊內選擇一個隨機位置
                random_h = np.random.randint(h_start, h_end)
                random_w = np.random.randint(w_start, w_end)

                # 將選中的位置設為未遮罩（1）
                mask_matrix[0, random_h, random_w, :] = 1

                w_start += block_size
            h_start += block_size

        # 將 mask_matrix 應用到所有幀
        mask = mask_matrix.repeat(T, 1, 1, 1)
        
    elif mask_type == 'fi':
        # 創建一個 4D mask (T, H, W, C)
        mask_matrix = torch.zeros((1, H, W, C), dtype=torch.float32)
        chosen_interval = np.random.choice(interval)
        # print(chosen_interval)
        for t in range(0, T, chosen_interval + 1):
            mask[t, :, :, :] = 1  # 遮蔽幀
        

    elif mask_type == 'nowcasting':
        # 創建一個 4D mask (T, H, W, C)
        mask = torch.ones((T, H, W, C), dtype=torch.float32)
        for t in range(keep, T):
            mask[t, :, :, :] = 0  # 遮蔽幀


    elif mask_type == 'stin':
        # 創建一個 4D mask (T, H, W, C)
        mask = torch.ones((T, H, W, C), dtype=torch.float32)
 
        # 對非保留的幀應用 sti 的遮罩方法
        for t in range(keep,T):
            mask_matrix = torch.zeros((1, H, W, C), dtype=torch.float32)
            block_size = np.random.choice(block_sizes)
            h_start = 0
            while h_start < H:
                w_start = 0
                while w_start < W:
                    h_end = min(h_start + block_size, H)
                    w_end = min(w_start + block_size, W)
                    random_h = np.random.randint(h_start, h_end)
                    random_w = np.random.randint(w_start, w_end)
                    mask_matrix[0, random_h, random_w, :] = 1
                    w_start += block_size
                h_start += block_size
            mask = mask_matrix.repeat(T, 1, 1, 1)

        for t in range(keep):
            mask[t, :, :, :] = 1  # 保留幀

    elif mask_type == 'stis' and mask_file is not None:
        # Load mask from file
        with open(mask_file, 'r') as f:
            mask_matrix = np.loadtxt(f)
        mask_matrix = torch.tensor(mask_matrix, dtype=torch.bool).unsqueeze(-1)  # Add an extra dimension

        # Ensure the mask matrix has the correct shape
        if mask_matrix.shape != (H, W, 1):
            raise ValueError(f"Mask matrix in {mask_file} does not match video spatial dimensions {H}x{W}")


        # Apply the same mask to all channels and frames
        for t in range(T):
            mask[t, :, :, :] = mask_matrix

    else:
        raise ValueError("Invalid mask type or mask file not provided for 'selfdefine' mask.")

    return mask

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """Dataset wrapper that支援 mp4/avi/h5 並依配置產生遮罩."""
        self.video_folder = args['data_root']
        self.transform = transform
        self.video_files = sorted(
            [
                os.path.join(self.video_folder, f)
                for f in os.listdir(self.video_folder)
                if f.endswith(('.mp4', '.avi', '.h5'))
            ],
            key=lambda f: extract_number(os.path.basename(f))
        )

        mask_cfg = args.get('mask', {})
        self.mask_type = mask_cfg.get('type', 'sti')
        self.mask_file = mask_cfg.get('file')
        self.block_sizes = mask_cfg.get('block_sizes', [4])
        self.mask_keep = mask_cfg.get('keep', 4)
        self.mask_interval = mask_cfg.get('interval', [2, 5])

        self.width = args['w']
        self.height = args['h']
        self.sample_length = args.get('sample_length')

        # print(f"Number of files in data_root1: {len(self.video_files)}")
        # print(f"Number of files in data_root2: {len(self.video2_files)}")
        # print(f"First file in data_root1: {self.video_files[0]}")
        # print(f"First file in data_root2: {self.video2_files[0]}")
        # ... 其他代碼保持不變 ...

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if idx >= len(self.video_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.video_files)} samples.")

        return self.process_file(self.video_files[idx])

    def process_file(self, file_path):
        if file_path.endswith(('.mp4', '.avi')):
            return self.process_video(file_path)
        elif file_path.endswith('.h5'):
            return self.process_hdf5(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def process_video(self, video_path):
        # 原有的 process_video 代碼
        vr = VideoReader(video_path)
        video_data = vr.get_batch(range(len(vr))).asnumpy()
        return self.post_process(video_data)

    def process_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            video_data = f['frames'][:]
     
            # 確保數據格式正確
            if video_data.ndim == 3:  # 如果是 (T, H, W)
                video_data = video_data[..., np.newaxis]  # 變成 (T, H, W, 1)
        return self.post_process(video_data)

    def post_process(self, video_data):
        # 維持原時間長度或根據 sample_length 截斷
        if self.sample_length is not None:
            T = min(self.sample_length, video_data.shape[0])
            video_data = video_data[:T]
        
        video_data = video_data.astype(np.float32) / 255.0
        if video_data.shape[-1] == 3:
            video_data = np.mean(video_data, axis=-1, keepdims=True)
        if self.transform:
            video_data = self.transform(video_data)

        mask = create_mask(
            video_data,
            mask_type=self.mask_type,
            mask_file=self.mask_file,
            block_sizes=self.block_sizes,
            keep=self.mask_keep,
            interval=self.mask_interval,
        )
        masked_video = video_data.clone()
        masked_video = masked_video * mask

        video_data = self._crop_center(video_data)
        masked_video = self._crop_center(masked_video)
        mask = self._crop_center(mask)
        return video_data, masked_video, mask

    def _crop_center(self, data):
        if data.shape[1] == self.height and data.shape[2] == self.width:
            return data
        old_height, old_width = data.shape[1], data.shape[2]
        start_x = max((old_width - self.width) // 2, 0)
        start_y = max((old_height - self.height) // 2, 0)
        end_y = start_y + self.height
        end_x = start_x + self.width
        return data[:, start_y:end_y, start_x:end_x, :]
