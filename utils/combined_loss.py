from torch import nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, coeficients):
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.coeficients = coeficients

    def forward(self, logits, true):
        result = 0
        
        for coef, loss in zip(self.coeficients, self.losses):
            result += coef * loss(logits, true)

        return result