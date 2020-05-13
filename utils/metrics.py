from utils.databunch import get_classes

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


def acc_cityscapes(input, target):
    target = target.squeeze(1)
    mask = (target != unlabeled) & (target != ego_v) & (target != rectification) & (target != roi) & (
                target != static) & (target != dynamic) & (target != ground) & (target != license)
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()
