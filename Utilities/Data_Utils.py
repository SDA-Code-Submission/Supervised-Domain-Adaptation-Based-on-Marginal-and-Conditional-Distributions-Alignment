from office31 import office31
from mnistusps import mnistusps
from Utilities.Digits_Utils import *
from Utilities.Office_Utils import *
from Utilities.Visda_Utils import *
from Utilities.CityCam_Utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
# from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
# from sklearn.model_selection import train_test_split
# import copy


##
def get_datasets(hp):
    if hp.Src in ['M', 'U']:
        digits_root = os.path.join('Datasets', 'digits')
        train, val, test = mnistusps(
            source_name="mnist" if hp.Src == 'M' else "usps",
            target_name="mnist" if hp.Tgt == 'M' else "usps",
            seed=np.random.randint(100),
            num_source_per_class=200,
            num_target_per_class=hp.SamplesPerClass,
            same_to_diff_class_ratio=3,
            image_resize=(16, 16),
            group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
            framework_conversion="pytorch",
            data_path=digits_root,  # downloads to "~/data" per default
        )
        train_dataset = MyDigits(train, transform=None)
        test_dataset = MyDigits(test, transform=None)
        val_dataset = MyDigits(val, transform=None)
        n_classes = 10

    elif hp.Src in ['A', 'W', 'D']:
        mapper = {'A': "amazon", 'W': "webcam", 'D': "dslr"}
        download_list = [
            ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
            ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
            ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
            ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
        ]
        office_root = os.path.join('Datasets', 'office31')
        if not (os.path.exists(office_root)):
            list(map(lambda args: download_data(office_root, *args), download_list))
        else:
            if not ([check_exits(office_root, x[0]) for x in download_list].count(None) == 4):
                list(map(lambda args: download_data(office_root, *args), download_list))
        train, val, test = office31(
            source_name=mapper[hp.Src],
            target_name=mapper[hp.Tgt],
            seed=np.random.randint(100),
            same_to_diff_class_ratio=3,
            image_resize=(240, 240),
            group_in_out=True,  # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
            framework_conversion="pytorch",
            office_path=office_root,
        )
        train_dataset = MyOffice(train, transform=office_train_transform)
        test_dataset = MyOffice(test, transform=office_val_transform)
        val_dataset = MyOffice(val, transform=office_val_transform)
        n_classes = 31
    elif hp.Src in ['S']:
        if not (os.path.exists(os.path.join('Datasets', 'visda-c'))):
            download_visda()
        train_folder = os.path.join('Datasets', 'visda-c', 'train')
        val_folder = os.path.join('Datasets', 'visda-c', 'validation')
        create_image_list(train_folder, 'my_source_image_list.txt')
        create_image_list(val_folder, 'my_target_image_list.txt')
        n_classes = len(VISDA_CLASSES)

        source_dataset = ImageList(root=train_folder,
                                   classes=VISDA_CLASSES,
                                   transform=visda_transform,
                                   data_list_file=os.path.join(os.getcwd(), train_folder,
                                                               'my_source_image_list.txt'))

        target_dataset = ImageList(root=val_folder,
                                   classes=VISDA_CLASSES,
                                   transform=visda_transform,
                                   data_list_file=os.path.join(os.getcwd(), val_folder,
                                                               'my_target_image_list.txt'))

        train_src_dataset, test_src_dataset = split_source_dataset(source_dataset, test_size=0.33, random_state=42)
        train_tgt_dataset, test_tgt_dataset = split_target_dataset(target_dataset, hp.SamplesPerClass)
    elif hp.Src.startswith('CityCam'):
        tgt_camera=hp.Tgt[8:]
        all_cameras=['253', '511', '572','495']
        dataset_dir = os.path.join('Datasets','CityCam')
        src_dataset = WCityCam(root_dir=dataset_dir, domains_list=[x for x in all_cameras if not(x==tgt_camera)])
        tgt_dataset = WCityCam(root_dir=dataset_dir, domains_list=[tgt_camera])
        train_src_dataset,test_src_dataset,train_tgt_dataset,test_tgt_dataset=partition_citycam_datasets(hp,src_dataset,tgt_dataset)
        n_classes = -1

    if hp.Src in ['M', 'U','A', 'W', 'D']:
        [train_loader, test_loader, val_loader] = [DataLoader(dataset, batch_size=hp.BatchSize,
                                                              shuffle=True, num_workers=0, drop_last=False)
                                                   for dataset in [train_dataset, test_dataset, val_dataset]]
    else:
        train_loader={'Src':DataLoader(train_src_dataset, batch_size=hp.BatchSize,
                                       shuffle=True, num_workers=0, drop_last=False),
                       'Tgt':DataLoader(train_tgt_dataset, batch_size=hp.BatchSize,
                                       shuffle=True, num_workers=0, drop_last=False)}
        test_loader = {'Src': DataLoader(test_src_dataset, batch_size=hp.BatchSize,
                                          shuffle=True, num_workers=0, drop_last=False),
                        'Tgt': DataLoader(test_tgt_dataset, batch_size=hp.BatchSize,
                                          shuffle=True, num_workers=0, drop_last=False)}
        val_loader = {'Src': DataLoader(test_src_dataset, batch_size=hp.BatchSize,
                                         shuffle=True, num_workers=0, drop_last=False),
                       'Tgt': DataLoader(test_tgt_dataset, batch_size=hp.BatchSize,
                                         shuffle=True, num_workers=0, drop_last=False)}
    return train_loader, test_loader, val_loader, n_classes


##
def download_data(root: str, file_name: str, archive_name: str, url_link: str):
    """
    Download file from internet url link.

    Args:
        root (str) The directory to put downloaded files.
        file_name: (str) The name of the unzipped file.
        archive_name: (str) The name of archive(zipped file) downloaded.
        url_link: (str) The url link to download data.

    .. note::
        If `file_name` already exists under path `root`, then it is not downloaded again.
        Else `archive_name` will be downloaded from `url_link` and extracted to `file_name`.
    """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Downloading {}".format(file_name))
        # if os.path.exists(os.path.join(root, archive_name)):
        #     os.remove(os.path.join(root, archive_name))
        try:
            download_and_extract_archive(url_link, download_root=root, filename=archive_name, remove_finished=False)
        except Exception:
            print("Fail to download {} from url link {}".format(archive_name, url_link))
            print('Please check you internet connection.'
                  "Simply trying again may be fine.")
            exit(0)

def check_exits(root: str, file_name: str):
    """Check whether `file_name` exists under directory `root`. """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Dataset directory {} not found under {}".format(file_name, root))
        # exit(-1)


##
def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to
    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

##
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

class ForeverDualBatchesIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, batch_size, device=None):
        self.iter = ForeverDataIterator(data_loader, device)
        self.batch_size = batch_size

    def __next__(self):
        data = next(self.iter)
        while not (data[0].shape[0] == self.batch_size):
            ex_data = next(self.iter)
            data[0] = torch.cat((data[0], ex_data[0]), dim=0)
            data[1] = torch.cat((data[1], ex_data[1]), dim=0)
            if data[0].shape[0] > self.batch_size:
                data[0] = data[0][:self.batch_size]
                data[1] = data[1][:self.batch_size]
        while not (data[2].shape[0] == self.batch_size):
            ex_data = next(self.iter)
            data[2] = torch.cat((data[2], ex_data[2]), dim=0)
            data[3] = torch.cat((data[3], ex_data[3]), dim=0)
            if data[2].shape[0] > self.batch_size:
                data[2] = data[2][:self.batch_size]
                data[3] = data[3][:self.batch_size]
        # print(not (data[0].shape[0] == self.batch_size) or not(data[1].shape[0] == self.batch_size))
        return data

    def __len__(self):
        return len(self.iter.data_loader)

class ForeverBatchesIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, batch_size, device=None):
        self.iter = ForeverDataIterator(data_loader, device)
        self.batch_size = batch_size

    def __next__(self):
        data = next(self.iter)
        while not (data[0].shape[0] == self.batch_size):
            ex_data = next(self.iter)
            data[0] = torch.cat((data[0], ex_data[0]), dim=0)
            data[1] = torch.cat((data[1], ex_data[1]), dim=0)
            if data[0].shape[0] > self.batch_size:
                data[0] = data[0][:self.batch_size]
                data[1] = data[1][:self.batch_size]
        return data

    def __len__(self):
        return len(self.iter.data_loader)

class DualIterator:
    def __init__(self, src_data_loader: DataLoader,tgt_data_loader: DataLoader,device=None):
        self.src_iter = ForeverBatchesIterator(src_data_loader,src_data_loader.batch_size)
        self.tgt_iter = ForeverBatchesIterator(tgt_data_loader, tgt_data_loader.batch_size)

    def __next__(self):
        src_data = next(self.src_iter)
        tgt_data = next(self.tgt_iter)
        return src_data[0],src_data[1],tgt_data[0],tgt_data[1]

def get_iterator(dataloader):
    if type(dataloader) is dict:
        return DualIterator(dataloader['Src'], dataloader['Tgt'])
    else:
        return ForeverDualBatchesIterator(dataloader,dataloader.batch_size)


##
#
# #!/usr/bin/python3
# import re
# import os
# import urllib.request
# import signal
# import argparse
# import json
# import sys
# from colorama import Fore, Style, init
#
# init()
#
# # this ANSI code lets us erase the current line
# ERASE_LINE = "\x1b[2K"
#
# COLOR_NAME_TO_CODE = {"default": "", "red": Fore.RED, "green": Style.BRIGHT + Fore.GREEN}
#
#
# def print_text(text, color="default", in_place=False, **kwargs):  # type: (str, str, bool, any) -> None
#     """
#     print text to console, a wrapper to built-in print
#
#     :param text: text to print
#     :param color: can be one of "red" or "green", or "default"
#     :param in_place: whether to erase previous line and print in place
#     :param kwargs: other keywords passed to built-in print
#     """
#     if in_place:
#         print("\r" + ERASE_LINE, end="")
#     print(COLOR_NAME_TO_CODE[color] + text + Style.RESET_ALL, **kwargs)
#
#
# def create_url(url):
#     """
#     From the given url, produce a URL that is compatible with Github's REST API. Can handle blob or tree paths.
#     """
#     repo_only_url = re.compile(r"https:\/\/github\.com\/[a-z\d](?:[a-z\d]|-(?=[a-z\d])){0,38}\/[a-zA-Z0-9]+$")
#     re_branch = re.compile("/(tree|blob)/(.+?)/")
#
#     # Check if the given url is a url to a GitHub repo. If it is, tell the
#     # user to use 'git clone' to download it
#     if re.match(repo_only_url,url):
#         print_text("✘ The given url is a complete repository. Use 'git clone' to download the repository",
#                    "red", in_place=True)
#         sys.exit()
#
#     # extract the branch name from the given url (e.g master)
#     branch = re_branch.search(url)
#     download_dirs = url[branch.end():]
#     api_url = (url[:branch.start()].replace("github.com", "api.github.com/repos", 1) +
#               "/contents/" + download_dirs + "?ref=" + branch.group(2))
#     return api_url, download_dirs
#
#
# def download(repo_url, flatten=False, output_dir="./"):
#     """ Downloads the files and directories in repo_url. If flatten is specified, the contents of any and all
#      sub-directories will be pulled upwards into the root folder. """
#
#     # generate the url which returns the JSON data
#     api_url, download_dirs = create_url(repo_url)
#
#     # To handle file names.
#     if not flatten:
#         if len(download_dirs.split(".")) == 0:
#             dir_out = os.path.join(output_dir, download_dirs)
#         else:
#             dir_out = os.path.join(output_dir, "/".join(download_dirs.split("/")[:-1]))
#     else:
#         dir_out = output_dir
#
#     try:
#         opener = urllib.request.build_opener()
#         opener.addheaders = [('User-agent', 'Mozilla/5.0')]
#         urllib.request.install_opener(opener)
#         response = urllib.request.urlretrieve(api_url)
#     except KeyboardInterrupt:
#         # when CTRL+C is pressed during the execution of this script,
#         # bring the cursor to the beginning, erase the current line, and dont make a new line
#         print_text("✘ Got interrupted", "red", in_place=True)
#         sys.exit()
#
#     if not flatten:
#         # make a directory with the name which is taken from
#         # the actual repo
#         os.makedirs(dir_out, exist_ok=True)
#
#     # total files count
#     total_files = 0
#
#     with open(response[0], "r") as f:
#         data = json.load(f)
#         # getting the total number of files so that we
#         # can use it for the output information later
#         total_files += len(data)
#
#         # If the data is a file, download it as one.
#         if isinstance(data, dict) and data["type"] == "file":
#             try:
#                 # download the file
#                 opener = urllib.request.build_opener()
#                 opener.addheaders = [('User-agent', 'Mozilla/5.0')]
#                 urllib.request.install_opener(opener)
#                 urllib.request.urlretrieve(data["download_url"], os.path.join(dir_out, data["name"]))
#                 # bring the cursor to the beginning, erase the current line, and dont make a new line
#                 print_text("Downloaded: " + Fore.WHITE + "{}".format(data["name"]), "green", in_place=True)
#
#                 return total_files
#             except KeyboardInterrupt:
#                 # when CTRL+C is pressed during the execution of this script,
#                 # bring the cursor to the beginning, erase the current line, and dont make a new line
#                 print_text("✘ Got interrupted", 'red', in_place=False)
#                 sys.exit()
#
#         for file in data:
#             file_url = file["download_url"]
#             file_name = file["name"]
#             file_path = file["path"]
#
#             if flatten:
#                 path = os.path.basename(file_path)
#             else:
#                 path = file_path
#             dirname = os.path.dirname(path)
#
#             if dirname != '':
#                 os.makedirs(os.path.dirname(path), exist_ok=True)
#             else:
#                 pass
#
#             if file_url is not None:
#                 try:
#                     opener = urllib.request.build_opener()
#                     opener.addheaders = [('User-agent', 'Mozilla/5.0')]
#                     urllib.request.install_opener(opener)
#                     # download the file
#                     urllib.request.urlretrieve(file_url, path)
#
#                     # bring the cursor to the beginning, erase the current line, and dont make a new line
#                     print_text("Downloaded: " + Fore.WHITE + "{}".format(file_name), "green", in_place=False, end="\n",
#                                flush=True)
#
#                 except KeyboardInterrupt:
#                     # when CTRL+C is pressed during the execution of this script,
#                     # bring the cursor to the beginning, erase the current line, and dont make a new line
#                     print_text("✘ Got interrupted", 'red', in_place=False)
#                     sys.exit()
#             else:
#                 download(file["html_url"], flatten, download_dirs)
#
#     return total_files
#
#
# # def main():
# #     if sys.platform != 'win32':
# #         # disbale CTRL+Z
# #         signal.signal(signal.SIGTSTP, signal.SIG_IGN)
# #
# #     parser = argparse.ArgumentParser(description="Download directories/folders from GitHub")
# #     parser.add_argument('urls', nargs="+",
# #                         help="List of Github directories to download.")
# #     parser.add_argument('--output_dir', "-d", dest="output_dir", default="./",
# #                         help="All directories will be downloaded to the specified directory.")
# #
# #     parser.add_argument('--flatten', '-f', action="store_true",
# #                         help='Flatten directory structures. Do not create extra directory and download found files to'
# #                              ' output directory. (default to current directory if not specified)')
# #
# #     args = parser.parse_args()
# #
# #     flatten = args.flatten
# #     for url in args.urls:
# #         total_files = download(url, flatten, args.output_dir)
# #
# #     print_text("✔ Download complete", "green", in_place=True)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# url='https://github.com/antoinedemathelin/wann/tree/master/dataset/citycam/253'
# download(url, False, os.path.join('Datasets','CityCam'))