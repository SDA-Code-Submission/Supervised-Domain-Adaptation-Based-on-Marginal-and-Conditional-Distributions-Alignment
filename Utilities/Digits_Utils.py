import torch
import torchvision.datasets as datasets
import numpy as np

class MyDigits(datasets.VisionDataset):

    def __init__(self, rect_loader, transform=None):
        super(MyDigits, self).__init__(rect_loader, transform=transform)

        self.train = rect_loader

    def __getitem__(self, index):
        src_img, src_label, tgt_img, tgt_label = self.train[index][0][0], self.train[index][1][0], \
                                                 self.train[index][0][1], self.train[index][1][1]
        src_img = np.array(src_img)
        src_img = torch.tensor(src_img).unsqueeze(0)
        tgt_img = np.array(tgt_img)
        tgt_img = torch.tensor(tgt_img).unsqueeze(0)
        if self.transform is not None:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)

        return src_img, src_label, tgt_img, tgt_label,

    def __len__(self):
        return len(self.train)
