
import torch
import numpy as np


def apply_cutmix_sprinkles(last_input, last_target, beta, num_holes = 8):
    new_input = last_input
    new_target = last_target

    for i in range(num_holes):
        λ = np.random.beta(beta, beta)
        λ = 1- max(λ, 1 - λ) * 0.16
        shuffle = torch.randperm(new_target.size(0)).to(new_input.device)
        # Get new input
        bbx1, bby1, bbx2, bby2 = rand_bbox(last_input.size(), λ)
        new_input = new_input.clone()
        new_input[:, ..., bby1:bby2, bbx1:bbx2] = last_input[shuffle, ..., bby1:bby2, bbx1:bbx2]
        new_target = new_target.clone()
        new_target[:, ..., bby1:bby2, bbx1:bbx2] = last_target[shuffle, ..., bby1:bby2, bbx1:bbx2]
    return new_input, new_target

def rand_bbox(last_input_size, λ):
    '''lambd is always between .5 and 1'''

    W = last_input_size[-1]
    H = last_input_size[-2]
    cut_rat = np.sqrt(1. - λ) # 0. - .707
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
