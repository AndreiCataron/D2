import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        #augmentarea datelor
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)


        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample