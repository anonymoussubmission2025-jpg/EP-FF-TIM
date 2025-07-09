import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_one_hot(y_s):
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


def get_loss(logits_s, logits_q, labels_s, lambdaa):
    Q = logits_q.softmax(2)
    y_s_one_hot = get_one_hot(labels_s)
    ce_sup = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1)  # Taking the mean over samples within a task, and summing over all samples
    ent_q = get_entropy(Q)
    cond_ent_q = get_cond_entropy(Q)
    loss = - (ent_q - cond_ent_q) + lambdaa * ce_sup
    return loss


def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)
    return q_cond_ent


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)

    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
        
    if callwrap:
        model_dict_load = model.state_dict()
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)
        
    else:
        model_dict_load = model.module.state_dict()
        model_dict_load.update(state)
        model.module.load_state_dict(model_dict_load)

def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm

def shuffle_list(x, y, z, oriLabel):
    temp = list(zip(y, z, x, oriLabel))

    random.shuffle(temp)

    y, z, x, oriLabel = zip(*temp)

    y = torch.stack(y, dim=0)
    z = torch.stack(z, dim=0)
    x = torch.stack(x, dim=0)
    oriLabel = torch.stack(oriLabel, dim=0)

    return x, y, z, oriLabel

def from_tensor_to_image(x):
    return x.view(3, 84, 84).permute(1, 2, 0) # [image_dim, image_dim, 3]

def get_dataset_labels(dataset_name):
    if dataset_name == 'mini_imagenet':
      return ['n01930112', 'n01981276', 'n02099601', 'n02110063', 'n02110341', 'n02116738', 'n02129165', 'n02219486', 'n02443484', 'n02871525', 'n03127925', 'n03146219', 'n03272010', 'n03544143', 'n03775546', 'n04146614', 'n04149813', 'n04418357', 'n04522168', 'n07613480']

def get_dataset_label_name(dataset_name, label):
    if dataset_name == 'mini_imagenet':
        mini_imagenet_dict = {
        "n01930112": "nematode, nematode worm, roundworm",
        "n01981276": "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
        "n02099601": "golden retriever",
        "n02110063": "malamute, malemute, Alaskan malamute",
        "n02110341": "dalmatian, coach dog, carriage dog",
        "n02116738": "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
        "n02129165": "lion, king of beasts, Panthera leo",
        "n02219486": "ant, emmet, pismire",
        "n02443484": "black-footed ferret, ferret, Mustela nigripes",
        "n02871525": "bookshop, bookstore, bookstall",
        "n03127925": "crate",
        "n03146219": "cuirass",
        "n03272010": "electric guitar",
        "n03544143": "hourglass",
        "n03775546": "mixing bowl",
        "n04146614": "school bus",
        "n04149813": "scoreboard",
        "n04418357": "theater curtain, theatre curtain",
        "n04522168": "vase",
        "n07613480": "trifle"  
        }

        return mini_imagenet_dict[label]

    if dataset_name == 'tiered_imagenet':
        tiered_imagenet = {

        }
        return

    if dataset_name == 'cub':
        cub = {

        }
        return

def get_labelToOriLabel_dict(label, oriLabel):
    label_dict = []

    for (a, b) in zip(label, oriLabel): 
        c = {a[i].item(): b[i].item() for i in range(len(a))} # for each task,mapping label->true_label
                                                              # eg: if [0,2,4,3,1] -> [12,14,1,3,5], then 0:12, 2:14, 4:1,...
        label_dict.append(c)

    return label_dict