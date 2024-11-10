import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class HighLowResDataset_test(Dataset):
    def __init__(self, high_res_dir_list,transform=None):
        """
        Args:
            high_res_dir (str): Directory with high-resolution images.
            low_res_dir (str): Directory with low-resolution images.
            transform (callable, optional): A function/transform to apply to both high and low-resolution images.
        """
        self.high_res_dir_list = high_res_dir_list
        self.transform = transform

        
        # if len(self.high_res_dir_list) != len(self.low_res_files):
        #     raise ValueError("The number of high-resolution and low-resolution images must be the same.")
    
    def __len__(self):
        return len(self.high_res_dir_list)
    
    def __getitem__(self, idx):
        high_res_file = self.high_res_dir_list[idx]
        high_res_image = Image.open(high_res_file).convert('RGB')
        width, height = high_res_image.size
        quarter_width = width // 4
        quarter_height = height // 4

        # Define the 16 regions in a 4x4 grid
        top_left_1 = high_res_image.crop((0, 0, quarter_width, quarter_height))
        top_left_2 = high_res_image.crop((quarter_width, 0, 2 * quarter_width, quarter_height))
        top_left_3 = high_res_image.crop((2 * quarter_width, 0, 3 * quarter_width, quarter_height))
        top_left_4 = high_res_image.crop((3 * quarter_width, 0, width, quarter_height))

        top_right_1 = high_res_image.crop((0, quarter_height, quarter_width, 2 * quarter_height))
        top_right_2 = high_res_image.crop((quarter_width, quarter_height, 2 * quarter_width, 2 * quarter_height))
        top_right_3 = high_res_image.crop((2 * quarter_width, quarter_height, 3 * quarter_width, 2 * quarter_height))
        top_right_4 = high_res_image.crop((3 * quarter_width, quarter_height, width, 2 * quarter_height))

        bottom_left_1 = high_res_image.crop((0, 2 * quarter_height, quarter_width, 3 * quarter_height))
        bottom_left_2 = high_res_image.crop((quarter_width, 2 * quarter_height, 2 * quarter_width, 3 * quarter_height))
        bottom_left_3 = high_res_image.crop((2 * quarter_width, 2 * quarter_height, 3 * quarter_width, 3 * quarter_height))
        bottom_left_4 = high_res_image.crop((3 * quarter_width, 2 * quarter_height, width, 3 * quarter_height))

        bottom_right_1 = high_res_image.crop((0, 3 * quarter_height, quarter_width, height))
        bottom_right_2 = high_res_image.crop((quarter_width, 3 * quarter_height, 2 * quarter_width, height))
        bottom_right_3 = high_res_image.crop((2 * quarter_width, 3 * quarter_height, 3 * quarter_width, height))
        bottom_right_4 = high_res_image.crop((3 * quarter_width, 3 * quarter_height, width, height))

        # Apply transformations if they exist
        if self.transform:
            top_left_1 = self.transform(top_left_1)
            top_left_2 = self.transform(top_left_2)
            top_left_3 = self.transform(top_left_3)
            top_left_4 = self.transform(top_left_4)
            
            top_right_1 = self.transform(top_right_1)
            top_right_2 = self.transform(top_right_2)
            top_right_3 = self.transform(top_right_3)
            top_right_4 = self.transform(top_right_4)
            
            bottom_left_1 = self.transform(bottom_left_1)
            bottom_left_2 = self.transform(bottom_left_2)
            bottom_left_3 = self.transform(bottom_left_3)
            bottom_left_4 = self.transform(bottom_left_4)
            
            bottom_right_1 = self.transform(bottom_right_1)
            bottom_right_2 = self.transform(bottom_right_2)
            bottom_right_3 = self.transform(bottom_right_3)
            bottom_right_4 = self.transform(bottom_right_4)

        # Stack the 16 images into a single tensor
        single_image = torch.stack((
            top_left_1, top_left_2, top_left_3, top_left_4,
            top_right_1, top_right_2, top_right_3, top_right_4,
            bottom_left_1, bottom_left_2, bottom_left_3, bottom_left_4,
            bottom_right_1, bottom_right_2, bottom_right_3, bottom_right_4
        ))

        return single_image
    

class HighLowResDataset_normal(HighLowResDataset_test):
    def __init__(self, high_res_dir_list, transform=None):
        super().__init__(high_res_dir_list, transform)
    def __getitem__(self, idx):
        high_res_file = self.high_res_dir_list[idx]
        high_res_image = Image.open(high_res_file).convert('RGB')
        if self.transform:
            high_res_image = self.transform(high_res_image)
        return high_res_image
        
if __name__ == "__main__":
    empty = ['/media/mlr_lab/6E18DC183015F19C/Ashu/Ashutosh_Dataset/VLN/Docker_Base/Matterport3DSimulator/data/v1/scans/PX4nDJXEHrG/matterport_skybox_images/000d2cac6cfd4f07b56d159ef5658a08_skybox3_sami.jpg']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = HighLowResDataset_test(empty, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    print(next(iter(dataloader)).squeeze().size())
    