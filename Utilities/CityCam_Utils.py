import os
import numpy as np
import torch

##
def concat_files(domains, root_dir):
    for domain in domains:
        path = os.path.join(root_dir, domain)
        for r, d, f in os.walk(path):
            for direct in d:
                if "checkpoints" not in direct:
                    x_path = path + "/" + direct + "/" + "X.npy"
                    y_path = path + "/" + direct + "/" + "y.npy"

                    Xi = np.load(x_path)
                    yi = np.load(y_path)

                    if len(yi) != len(Xi):
                        print(len(yi), len(Xi))

                    try:
                        X = np.concatenate((X, Xi))
                        y = np.concatenate((y, yi))
                    except:
                        X = np.copy(Xi)
                        y = np.copy(yi)
    return X, y


class WCityCam():
    def __init__(self, root_dir, domains_list=['253', '511', '572', '495']):
        self.root_dir = root_dir
        self.X, self.y = concat_files(domains_list, root_dir)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


##
def partition_citycam_datasets(hp, src_dataset, tgt_dataset):
    n_smaples_src_train, n_samples_src_test = 5000, 5000
    n_smaples_tgt_train, n_samples_tgt_test = hp.SamplesPerClass, 2000

    [train_src_dataset, test_src_dataset, _] = torch.utils.data.random_split(src_dataset,
                                                                             [n_smaples_src_train, n_samples_src_test,
                                                                              len(src_dataset) - (
                                                                                          n_smaples_src_train + n_samples_src_test)])
    [train_tgt_dataset, test_tgt_dataset, _] = torch.utils.data.random_split(tgt_dataset,
                                                                             [n_smaples_tgt_train, n_samples_tgt_test,
                                                                              len(tgt_dataset) - (
                                                                                          n_smaples_tgt_train + n_samples_tgt_test)])
    return train_src_dataset, test_src_dataset, train_tgt_dataset, test_tgt_dataset
