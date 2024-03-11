## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


## CCSA
def ccsa_loss(x, y, class_eq):
    """
    taken from: https://github.com/YooJiHyeong/CCSA_PyTorch/blob/master/main.py
    a pytorch implementation of the official code https://github.com/samotiian/CCSA
    """
    margin = 1
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)  # if eqaul classes: penalize dist
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)  # if different - penalize if larger than margin
    return loss.mean()


## dSNE
def dsne_loss(fts, ys, ftt, yt):
    """
    taken from: https://github.com/aws-samples/d-SNE/blob/master/train_val/custom_layers.py
    """
    bs_tgt, bs_src = len(fts), len(ftt)
    embed_size = fts.shape[1]
    margin = 1
    fts = torch.nn.functional.normalize(fts, p=2, dim=1)
    ftt = torch.nn.functional.normalize(ftt, p=2, dim=1)

    fts_rpt = torch.broadcast_to(fts.unsqueeze(dim=0), size=(bs_tgt, bs_src, embed_size))
    ftt_rpt = torch.broadcast_to(ftt.unsqueeze(dim=1), size=(bs_tgt, bs_src, embed_size))
    dists = torch.sum(torch.square(ftt_rpt - fts_rpt), axis=2)
    yt_rpt = torch.broadcast_to(yt.unsqueeze(dim=1), size=(bs_tgt, bs_src)).type(torch.int32)
    ys_rpt = torch.broadcast_to(ys.unsqueeze(dim=0), size=(bs_tgt, bs_src)).type(torch.int32)

    y_same = ((yt_rpt - ys_rpt) == 0)
    y_diff = torch.logical_not(y_same)

    intra_cls_dists = dists * y_same
    inter_cls_dists = dists * y_diff
    (max_dists, _) = torch.max(dists, axis=1, keepdims=True)
    max_dists = max_dists.expand((bs_tgt, bs_src))
    revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)
    (max_intra_cls_dist, _) = torch.max(intra_cls_dists, axis=1)
    (min_inter_cls_dist, _) = torch.min(revised_inter_cls_dists, axis=1)
    # loss = torch.sum(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    loss = torch.mean(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    return loss


## NEM
def nem_loss(tgt_img, src_feature, tgt_feature, src_label, tgt_label):
    """
    implementation of NEM loss, the weight of the graph loss is taken according to figure 7 in
    https://ieeexplore.ieee.org/document/8836530
    """
    pairwise_distances_f = torch.cdist(tgt_feature, tgt_feature, compute_mode="donot_use_mm_for_euclid_dist")
    tmp = torch.flatten(tgt_img, start_dim=1)
    pairwise_distances_x = torch.cdist(tmp, tmp, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    # scale = get_scale_fac(d=pairwise_distances_x,discard_diag=False)
    scale = torch.median(pairwise_distances_x)
    input_affinity = torch.exp(-pairwise_distances_x / scale)
    loss_graph = torch.mean(input_affinity * pairwise_distances_f)

    gt_label_sim = (src_label.repeat(tgt_label.shape[0], 1).T == (tgt_label.repeat(src_label.shape[0], 1))).type(
        torch.uint8)
    pairwise_cross_distances = torch.cdist(src_feature, tgt_feature, compute_mode="donot_use_mm_for_euclid_dist")

    loss_contrastive = torch.mean(gt_label_sim * pairwise_cross_distances) + \
                       torch.mean((1 - gt_label_sim) * (1 - pairwise_cross_distances).clamp(min=0))

    loss = loss_contrastive + 0.001 * loss_graph
    return loss


## UDA - CORAL
def coral_loss(source, target):
    """
    From the paper, the vectors that compose Ds and Dt are D-dimensional vectors
    :param source: torch tensor: source data (Ds) with dimensions DxNs
    :param target: torch tensor: target data (Dt) with dimensons DxNt
    """
    d = source.size(1)  # d-dimensional vectors (same for source, target)
    source_covariance = compute_covariance(source)
    target_covariance = compute_covariance(target)
    loss = torch.norm(torch.mul((source_covariance - target_covariance),
                                (source_covariance - target_covariance)), p="fro")
    # loss=torch.norm(logm(source_covariance)- logm(target_covariance)) #log-euclidean norm
    loss = loss / (4 * d * d)
    return loss


def compute_covariance(data):
    """
    Compute covariance matrix for given dataset as shown in paper (eqs 2 and 3).
    :param data: torch tensor: input source/target data
    """
    # data dimensions: nxd (this for Ns or Nt)
    n = data.size(0)  # get batch size
    # check gpu or cpu support
    if data.is_cuda:
        device = data.device
    else:
        device = torch.device("cpu")
    # proper matrix multiplication for right side of equation (2)
    ones_vector = torch.ones(n).resize_(1, n).to(device=device)  # 1xN dimensional vector (transposed)
    one_onto_D = torch.mm(ones_vector, data)
    mult_right_terms = torch.mm(one_onto_D.t(), one_onto_D)
    mult_right_terms = torch.div(mult_right_terms, n)  # element-wise divison

    # matrix multiplication for left side of equation (2)
    mult_left_terms = torch.mm(data.t(), data)

    covariance_matrix = 1 / (n - 1) * torch.add(mult_left_terms, -1 * (mult_right_terms))

    return covariance_matrix


## UDA - MMD
def MMD(x, y, kernel, mmd_kernel_scale='Fixed', mmd_kernel_bandwidth=[0.1, 0.5, 1, 2]):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    nx = x.shape[0]
    ny = y.shape[0]
    dxx = torch.cdist(x, x) ** 2  # (x_i-x_j)^2
    dyy = torch.cdist(y, y) ** 2  # (y_i-y_j)^2
    dxy = torch.cdist(x, y) ** 2  # (x_i-y_j)^2
    device = x.device if x.is_cuda else torch.device("cpu")
    XX, YY, XY = (torch.zeros(dxx.shape).to(device),
                  torch.zeros(dyy.shape).to(device),
                  torch.zeros(dxy.shape).to(device))

    if kernel == "rbf":
        if mmd_kernel_scale == 'Fixed':
            for a in mmd_kernel_bandwidth:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)
        if mmd_kernel_scale == 'Median':
            for a in mmd_kernel_bandwidth:
                XX += torch.exp(-0.5 * dxx / (a * torch.median(dxx).item()))
                YY += torch.exp(-0.5 * dyy / (a * torch.median(dyy).item()))
                XY += torch.exp(-0.5 * dxy / (a * torch.median(dxy).item()))
        if mmd_kernel_scale == 'Auto':
            x_scale = get_scale_fac(dxx, discard_diag=True)
            y_scale = get_scale_fac(dyy, discard_diag=False)
            xy_scale = get_scale_fac(dxy, discard_diag=False)
            scale = torch.max(torch.max(x_scale, y_scale), xy_scale)

            for a in mmd_kernel_bandwidth:
                XX += torch.exp(-0.5 * dxx / (a * scale.item()))
                YY += torch.exp(-0.5 * dyy / (a * scale.item()))
                XY += torch.exp(-0.5 * dxy / (a * scale.item()))

    return (torch.sum(XX) / (nx ** 2) + torch.sum(YY) / (ny ** 2) - 2. * torch.sum(XY) / (nx * ny))


## UDA experiments (for further use)
# # MDD
# from dalib.adaptation.mdd import MarginDisparityDiscrepancy, ImageClassifier
# # DAN
# from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
# from dalib.modules.kernels import GaussianKernel
# # DANN
# from dalib.modules.domain_discriminator import DomainDiscriminator
# from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
# # CDAN
# from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
# from cdannn import ConditionalDomainAdversarialLoss, ImageClassifie

## Ours
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """
    Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``.
        """
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def get_scale_fac(d, discard_diag=False):
    """
    return the scale factor for the kernel regression
    when d=pdist(x,y), set discard_diag to True, when d=pdist(x,x), set discard_diag to False.
    """
    # when d=pdist(x,y), set discard_diag to True, when d=pdist(x,x), set discard_diag to False.
    device = d.device if d.is_cuda else torch.device("cpu")
    if discard_diag:
        [min_per_row, _] = torch.min(d + torch.max(d) * torch.eye(n=d.shape[0], m=d.shape[1]).to(device), dim=1)
    else:
        [min_per_row, _] = torch.min(d, dim=1)
    return torch.max(min_per_row)


def get_weight_matrix(pairwise_distances, n_nearest_neighbours=5, kernel_scale='Fixed', use_scale_fac_grad=True):
    """
    returns the kernel for kernel regression
    """
    scale_fac = torch.tensor(1.0)
    if n_nearest_neighbours > 0:
        _, indices = torch.sort(pairwise_distances)
        weight_matrix = torch.zeros_like(pairwise_distances)
        for k in range(n_nearest_neighbours):
            weight_matrix[range(weight_matrix.shape[0]), indices[:, k]] = (1 / n_nearest_neighbours)
        k_pairwise_distances = pairwise_distances * weight_matrix
        weight_matrix = torch.diag(1 / torch.sum(k_pairwise_distances, dim=1)) @ k_pairwise_distances
    else:
        if kernel_scale == 'Fixed':
            scale_fac = torch.tensor(1.0)
        if kernel_scale == 'Auto':
            scale_fac = get_scale_fac(pairwise_distances, discard_diag=True)
        if kernel_scale == 'Median':
            scale_fac = torch.median(pairwise_distances)
        if not use_scale_fac_grad:
            scale_fac = scale_fac.item()
        weight_matrix = nn.Softmax(dim=1)(-pairwise_distances / scale_fac)
    return weight_matrix


def get_one_hot_encoding(label, n_classes):
    one_hot_label = torch.zeros(len(label), n_classes).to(label.device)
    one_hot_label[range(len(label)), label] = 1
    return one_hot_label


def get_cdca_term(src_feature, tgt_feature, src_label, tgt_label, n_classes, criterion, hp):
    if n_classes > 0:
        src_labels_one_hot = get_one_hot_encoding(src_label, n_classes)
        tgt_labels_one_hot = get_one_hot_encoding(tgt_label, n_classes)
    else:
        src_labels_one_hot = src_label.reshape(-1, 1).float()
        tgt_labels_one_hot = tgt_label.reshape(-1, 1).float()

    pairwise_cross_distances = torch.cdist(tgt_feature, src_feature) ** 2
    # --- E_{Dt}[ f_s(x)-f_t(x)|] --- (predictions based on nearest neighbours from the source domain)
    weight_matrix_t2s = get_weight_matrix(pairwise_cross_distances, n_nearest_neighbours=hp.NumberOfNearestNeighbours,
                                          kernel_scale=hp.KernelScale,
                                          use_scale_fac_grad=False)
    labels_t_nn = weight_matrix_t2s @ src_labels_one_hot  # pred=E[f(s)]
    # ---E_{Ds}[|f_s(x)-f_t(x)|] --- (predictions based on nearest neighbours from the target domain)
    weight_matrix_s2t = get_weight_matrix(pairwise_cross_distances.T, n_nearest_neighbours=hp.NumberOfNearestNeighbours,
                                          kernel_scale=hp.KernelScale,
                                          use_scale_fac_grad=False)
    labels_s_nn = weight_matrix_s2t @ tgt_labels_one_hot  # pred=E[f(t)]
    #
    loss_t_nn = criterion(labels_t_nn, tgt_label)
    loss_s_nn = criterion(labels_s_nn, src_label)
    loss_cdca = torch.min(loss_s_nn, loss_t_nn)
    return loss_cdca


def get_uda_term(src_feature, tgt_feature, hp):
    if hp.UdaMethod == 'CORAL':
        loss_uda = coral_loss(src_feature, tgt_feature)
    if hp.UdaMethod == 'MMD':
        loss_uda = MMD(src_feature, tgt_feature,
                       kernel='rbf', mmd_kernel_scale=hp.KernelScale,
                       mmd_kernel_bandwidth=[0.1, 0.5, 1, 2])
    # if hp.UdaMethod == 'DAN':
    #     loss_uda = mkmmd_loss(src_feature, tgt_feature)
    # if hp.UdaMethod == 'DANN':
    #     loss_uda = domain_adv(src_feature, tgt_feature)
    #     domain_acc = domain_adv.domain_discriminator_accuracy
    # if hp.UdaMethod == 'CDAN':
    #     # loss_uda = domain_adv(src_feature, tgt_feature)
    #     loss_uda = domain_adv(src_pred, src_feature, tgt_pred, tgt_feature)
    # if hp.UdaMethod == 'MDD':
    #     loss_uda = -mdd(y_s, y_s_adv, y_t, y_t_adv)
    return loss_uda
