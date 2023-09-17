import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
def con_loss_new(features, labels):
    eps = 1e-6
    B= features.shape[0]
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    neg_label_matrix_new = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = 1 + cos_matrix
    margin = 0.3
    sim = (1 + cos_matrix) / 2.0
    scores = 1 - sim
    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
    mask = torch.eye(features.size(0)).cuda()
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
                (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
    relative_dis1 = margin + positive_scores - scores
    neg_label_matrix_new[relative_dis1 < 0] = 0
    neg_label_matrix = neg_label_matrix * neg_label_matrix_new

    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B * B

    return loss