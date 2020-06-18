import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import special
import random


def apply_cowmix(last_input, last_target, prob=0.5):
    shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
    shape = np.array(last_target.shape)[1:]

    mask, mask_inv = generate_mask(shape, prob)
    mask = torch.tensor(mask).cuda()
    mask_inv = torch.tensor(mask_inv).cuda()

    # Not sure yet if i need the clone().
    shuffled_input = last_input.clone()[shuffle]
    shuffled_target = last_target.clone()[shuffle]

    new_input = last_input * mask + shuffled_input * mask_inv
    new_target = last_target * mask + shuffled_target * mask_inv

    return new_input, new_target


def apply_cowout(last_input, last_target, prob=0.5):
    shape = np.array(last_target.shape)[1:]

    mask, mask_inv = generate_mask(shape, prob)
    mask = torch.tensor(mask).cuda()

    new_input = last_input * mask
    new_target = last_target * mask

    return new_input, new_target

def generate_mask(shape, prob):
    sigmas = list(range(8, 33))
    sigma = random.choice(sigmas)
    noise = random_noise(shape - 1)
    convolved = gaussian_filter(noise, sigma=sigma)

    mean = np.mean(convolved)
    std = np.std(convolved)

    threshold = special.erfinv(2*prob - 1) * np.sqrt(2) * std + mean
    mask = np.where(convolved <= threshold, 0, 1)
    mask_inv = np.where(convolved > threshold, 0, 1)
    return mask, mask_inv


def random_noise(shape, channels=1):
    result = np.random.normal(size=(channels,) + shape)
    return result
