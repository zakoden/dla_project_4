import torch
from torch import nn


class AllLoss(nn.Module):
    def __init__(self):
        super(AllLoss, self).__init__()

    def adv_discriminator_loss(self, logits_true, logits_pred):
        sum_loss = 0.0
        for ind in range(len(logits_true)):
            sum_loss += torch.mean((1 - logits_true[ind]) ** 2)
            sum_loss += torch.mean((logits_pred[ind]) ** 2)
        return sum_loss

    def adv_generator_loss(self, logits_pred):
        sum_loss = 0.0
        for ind in range(len(logits_pred)):
            sum_loss += torch.mean((1 - logits_pred[ind]) ** 2)
        return sum_loss

    def features_loss(self, features_true, features_pred):
        sum_loss = 0.0
        for outer_ind in range(len(features_true)):
            for inner_ind in range(len(features_true[outer_ind])):
                sum_loss += torch.mean(torch.abs(features_true[outer_ind][inner_ind] - features_pred[outer_ind][inner_ind]))
        return sum_loss

