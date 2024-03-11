import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


def get_acc(pred,gt):
    _, idx = pred.max(dim=1)
    acc = (idx == gt).sum().cpu().item() / len(idx)
    return acc

def perf_eval(hp,net, loader, BatchLim=np.inf, Domain='Tgt', Text=""):
    net.eval()
    metrics = []
    if type(loader) is dict:
        progress_bar = tqdm(enumerate(loader[Domain]), desc=Text, bar_format="{desc:20}{percentage:2.0f}{r_bar}")
    else:
        progress_bar = tqdm(enumerate(loader), desc=Text, bar_format="{desc:20}{percentage:2.0f}{r_bar}")
    for ind, z in progress_bar:
        if ind > BatchLim:
            break
        if type(loader) is dict:
            (img, label) = z
        else:
            if Domain == 'Tgt':
                img, label = z[2], z[3]
            else:
                img, label = z[0], z[1]
        img = img.to(net.device, dtype=torch.float)
        label = label.to(net.device, dtype=torch.long)
        pred = net(img)
        if hp.TaskObjective=='CE':
            acc=get_acc(pred, label)
            metrics.append(acc)
        else:
            metrics.append(nn.L1Loss()(pred.reshape(-1, 1).float(), label.reshape(-1, 1).float()).item())

    return metrics