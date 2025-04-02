import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, csv_file, transforms_MR_=None, transforms_PET_=None):
        self.transform_MR_ = transforms_MR_
        self.transform_PET_ = transforms_PET_

        # 从CSV文件读取数据
        self.dataframe = pd.read_csv(csv_file)
        self.files_A = self.dataframe['MR'].tolist()
        self.files_B = self.dataframe['PT'].tolist()

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index])
        image_B = Image.open(self.files_B[index])

        if image_A.mode != "L":
            image_A = image_A.convert("L")
        if image_B.mode != "L":
            image_B = image_B.convert("L")

        if self.transform_MR_:
            item_A = self.transform_MR_(image=np.array(image_A))['image']
            if not isinstance(item_A, torch.Tensor):
                item_A = transforms.ToTensor()(item_A)
        else:
            item_A = transforms.ToTensor()(image_A)

        if self.transform_PET_:
            item_B = self.transform_PET_(image=np.array(image_B))['image']
            if not isinstance(item_B, torch.Tensor):
                item_B = transforms.ToTensor()(item_B)
        else:
            item_B = transforms.ToTensor()(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return len(self.files_A)
