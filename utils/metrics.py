from utils.databunch import get_classes
import torch

classes = get_classes()
name2id = {v: k for k, v in enumerate(classes)}
unlabeled = name2id['unlabeled']
ego_v = name2id['ego vehicle']
rectification = name2id['rectification border']
roi = name2id['out of roi']
static = name2id['static']
dynamic = name2id['dynamic']
ground = name2id['ground']
license = name2id['license plate']

ignored =[unlabeled, ego_v, rectification, roi, static, dynamic, ground, license] 
non_ignored = [name2id[c] for c in classes if c not in ignored]
SMOOTH = 1e-6


def acc_cityscapes(input, target):
    target = target.squeeze(1)
    mask = (target != unlabeled) & (target != ego_v) & (target != rectification) & (target != roi) & (
                target != static) & (target != dynamic) & (target != ground) & (target != license)
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def mIOU(outputs: torch.Tensor, labels: torch.Tensor):

    outputs = outputs.argmax(dim=1)
    
    labels = labels.squeeze(1)
    miou= 0
    samples = 0
    for c in non_ignored:
        o  = outputs == c
        l = labels == c
        intersection = (o & l).float().sum((1, 2))  
        union = (o | l).float().sum((1, 2))         
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
          
        non_zero = (union>0).float()
        iou *=non_zero
        miou += iou.sum()
        samples+=non_zero.sum()
        
    miou /= samples
    
    return miou
