import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from .activations import relu15


class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        loss = self.loss(X, target)

        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size
        return loss


class ReLU15LossFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, X, target, tau):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]
        p_star = relu15(X, dim=-1, tau=tau)
        loss = (p_star.sum(dim=1) - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75
        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, X-2*tau)
        ctx.save_for_backward(p_star)
        return loss
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        p_star,  = ctx.saved_tensors 
        grad = grad_output.unsqueeze(1) * p_star
        ret = (grad,)

        # pad with as many Nones as needed
        return ret + (None,) * (1 + 2)

def relu15_loss(X, target, tau=0.0):
    return ReLU15LossFunction.apply(X, target, tau)


class ReLU15Loss(_GenericLoss):
    def __init__(self, ignore_index=-100, reduction='elementwise_mean', tau=0.0):
        super(ReLU15Loss, self).__init__(ignore_index, reduction)
        self.tau = tau

    def loss(self, X, target):
        return relu15_loss(X, target, self.tau)





