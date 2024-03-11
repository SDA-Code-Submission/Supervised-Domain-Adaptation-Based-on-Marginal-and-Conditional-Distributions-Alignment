# import torch
import torchvision.datasets as datasets
# import numpy as np
import torchvision.transforms as T
# import os
# from typing import Optional
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

##
class MyOffice(datasets.VisionDataset):
    def __init__(self, rect_loader, transform=None):
        super(MyOffice, self).__init__(rect_loader, transform=transform)

        self.train = rect_loader
        tmp = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator','desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
        self.mapper = {}
        for ik, k in enumerate(tmp):
            self.mapper[k] = ik

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        src_img, src_label, tgt_img, tgt_label = self.train[index][0][0], self.train[index][1][0],\
                                                 self.train[index][0][1], self.train[index][1][1]
        if self.transform is not None:
            src_img = self.transform(src_img)
        if self.transform is not None:
            tgt_img = self.transform(tgt_img)
        return src_img, self.mapper[src_label], tgt_img, self.mapper[tgt_label],

    def __len__(self):
        return len(self.train)


##
class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,          resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)

def get_val_transform(resizing='default', resize_size=224,        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

# class OfficeArgsClass():
#     def __init__(self, ):
#         self.data = 'Office31'
#         self.source = 'A'
#         self.target = 'W'
#         self.train_resizing = 'default'
#         # self.train_resizing='cen.crop'
#         self.val_resizing = 'default'
#         # self.train_resizing='cen.crop'
#         # self.val_resizing='cen.crop'
#         self.no_hflip = False
#         self.resize_size = 224
#         self.norm_mean = (0.485, 0.456, 0.406)
#         self.norm_std = (0.229, 0.224, 0.225)
#         self.root = ''
#         self.batch_size = 32
#         self.workers = 2
#         self.arch = 'resnet50'
#         self.scratch = False
#         self.no_pool = False
#         self.bottleneck_dim = 256
# args = OfficeArgsClass()

office_train_transform = get_train_transform('default',
                                             random_horizontal_flip=True,
                                             random_color_jitter=False,
                                             resize_size=224,
                                             norm_mean= (0.485, 0.456, 0.406),
                                             norm_std=(0.229, 0.224, 0.225))

office_val_transform = get_val_transform('default',
                                         resize_size=224,
                                         norm_mean=(0.485, 0.456, 0.406),
                                         norm_std=(0.229, 0.224, 0.225))




