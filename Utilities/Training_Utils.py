import torch
import torch.nn as nn

def get_objective(hp):
    criterion=-1
    if hp.TaskObjective == 'CE':
        criterion = nn.CrossEntropyLoss()
    if hp.TaskObjective == 'L2':
        criterion = nn.L1Loss()
    if hp.TaskObjective == 'L1':
        criterion = lambda x, y: nn.L1Loss()(x.reshape(-1, 1).float(), y.reshape(-1, 1).float())

    if criterion==-1:
        raise Exception("TaskObjective can be one of the followings:'CE','L1','L2'.")
    return criterion

def get_optimizer(net,hp):
    if hp.Optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(),
                                         lr=hp.LearningRate, weight_decay=hp.WD)
    if hp.Optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=hp.LearningRate, momentum=0.9, weight_decay=hp.WD, nesterov=True)
    elif hp.Optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=hp.LearningRate, weight_decay=hp.WD)
    return optimizer


