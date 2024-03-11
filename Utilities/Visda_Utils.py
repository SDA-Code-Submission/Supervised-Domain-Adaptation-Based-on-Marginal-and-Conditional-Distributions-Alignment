from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as torchdatasets
import torchvision.transforms as T
import os
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
import copy
import numpy as np
import time
import wget
import tarfile

##
VISDA_CLASSES = ['plant', 'train', 'bus', 'horse', 'knife', 'bicycle', 'aeroplane', 'person', 'motorcycle', 'car',
                 'skateboard', 'truck']


##
def download_and_extract(url, path):
    wget.download(url, path)
    my_tar = tarfile.open(path)
    my_tar.extractall(os.path.join('Datasets', 'visda-c'))  # specify which folder to extract to
    my_tar.close()
    os.remove(path)


def download_visda():
    """
    download visda-c dataset from here https: // github.com / VisionLearningGroup / taskcv - 2017 - public / tree / master / classification
    """
    start = time.time()
    print("Downloading VisDA-C dataset (this may take ~15 minutes)")
    val_path = os.path.join('Datasets', 'tmp-visda-c-validation.tar')
    val_url = "http://csr.bu.edu/ftp/visda17/clf/validation.tar"
    download_and_extract(val_url, val_path)
    train_path = os.path.join('Datasets', 'tmp-visda-c-train.tar')
    train_url = "http://csr.bu.edu/ftp/visda17/clf/train.tar"
    download_and_extract(train_url, train_path)
    end = time.time()
    print("Finished downloading, took %g seconds" % (end - start))


def create_image_list(dataset_folder, image_list_name='my_source_image_list.txt'):
    images_tuples = []
    for class_index, curr_class in enumerate(VISDA_CLASSES):
        curr_class_image_list = os.listdir(os.path.join(dataset_folder, curr_class))
        for img_name in curr_class_image_list:
            images_tuples.append(os.path.join(curr_class, img_name) + ' %g' % class_index)
            # curr_class+'/'+img_name+' %g'%class_index)

    with open(os.path.join(dataset_folder, image_list_name), 'w') as texy_file:
        texy_file.write('\n'.join(images_tuples))
    return


##
class ImageList(torchdatasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented


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


visda_transform = T.Compose([
    ResizeImage(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=0, std=1)
])


def split_source_dataset(source_dataset, test_size=0.33, random_state=42):
    train_list, test_list = train_test_split(source_dataset.samples, test_size=test_size, random_state=random_state)
    train_src_dataset = copy.deepcopy(source_dataset)
    test_src_dataset = copy.deepcopy(source_dataset)
    train_src_dataset.samples = train_list
    test_src_dataset.samples = test_list
    return train_src_dataset,test_src_dataset


def split_target_dataset(target_dataset,samples_per_class):
    train_list, test_list = [], []
    class_samples = np.array([x[1] for x in target_dataset.samples])
    for iclass in range(len(target_dataset.classes)):
        class_inds = np.where(class_samples == iclass)[0]
        class_inds = np.random.permutation(class_inds)
        test_list.extend([target_dataset.samples[x] for x in class_inds[samples_per_class:]])
        train_list.extend([target_dataset.samples[x] for x in class_inds[:samples_per_class]])
    train_tgt_dataset = copy.deepcopy(target_dataset)
    test_tgt_dataset = copy.deepcopy(target_dataset)
    train_tgt_dataset.samples = train_list
    test_tgt_dataset.samples = test_list
    return train_tgt_dataset, test_tgt_dataset

