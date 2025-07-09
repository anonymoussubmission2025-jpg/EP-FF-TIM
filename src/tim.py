import torch.nn.functional as F
from .utils import get_mi, get_cond_entropy, get_entropy, get_one_hot
from tqdm import tqdm
from sacred import Ingredient
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.utils import get_dataset_labels, get_dataset_label_name, get_labelToOriLabel_dict
from src.utils import load_pickle, save_pickle
from src.utils import UnNormalize
from src.utils import from_tensor_to_image
import matplotlib.pyplot as plt

writer = SummaryWriter()

tim_ingredient = Ingredient('tim')


@tim_ingredient.config
def config():
    temp = 15
    loss_weights = [0.1, 1.0, 0.1]  # [Xent, H(Y), H(Y|X)]
    lr = 1e-4
    iter = 150
    alpha = 1.0


class TIM(object):
    @tim_ingredient.capture
    def __init__(self, temp, loss_weights, iter, model):
        self.temp = temp
        self.loss_weights = loss_weights.copy()
        self.iter = iter
        self.model = model
        self.init_info_lists()

    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []
        self.pred_vs_actual_indexDifference = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2))
                            - 1 / 2 * (self.weights ** 2).sum(2).view(n_tasks, 1, -1)
                            - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))

        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def init_weights(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s,
                         y_q=y_q)
        self.model.train()

    def compute_lambda(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (
                1 + self.loss_weights[2]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        self.timestamps.append(new_time)
        self.mutual_infos.append(get_mi(probs=q_probs))
        self.entropy.append(get_entropy(probs=q_probs.detach()))
        self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach()))
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy,
                'acc': self.test_acc, 'losses': self.losses}

    def run_adaptation(self, support, query, y_s, y_q, callback):
        """
        Corresponds to the baseline (no transductive inference = SimpleShot)
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
            callback : VisdomLogger or None to plot in live our metrics


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        pass

    def run_visual_analysis(self, img_index, oriLabelIndex, raw_images_path, device):
        all_images = load_pickle(raw_images_path)
        dataset_labels = get_dataset_labels('mini_imagenet')
        labels_dict = get_labelToOriLabel_dict(label=self.actual_label, oriLabel=oriLabelIndex)

        false_cases = []
        for i, task in enumerate(self.pred_vs_actual_indexDifference):
            for j, pred in enumerate(task):
                if pred == False: #if predicted != actual
                    all_image_index = img_index[i, j].item() # the index of the image in the list of all images
                    case_label_dict = labels_dict[i]
                    case_predLabel = case_label_dict[ self.predicted_label[i, j].item() ] # convert to oriLabel index
                    case_actualLabel = case_label_dict[ self.actual_label[i, j].item() ] # convert to oriLabel index

                    false_case = {
                    'image_index': all_image_index,
                    'label_dict': case_label_dict,
                    'pred_label_name': get_dataset_label_name('mini_imagenet', dataset_labels[case_predLabel] ), # convert label index to name
                    'actual_label_name': get_dataset_label_name('mini_imagenet', dataset_labels[case_actualLabel] ) # convert label index to name
                    }
                    false_cases.append(false_case)

        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # unnormalize image with imagenet stats to recover original image before pre-processing
        
        for case in false_cases:
            img_index = case['image_index']
            img_tensor = from_tensor_to_image( unorm(all_images[img_index]) )
            pred_label = case['pred_label_name']
            actual_label = case['actual_label_name']

            print("*********")
            
            plt.axis('off')
            plt.imshow(img_tensor)
            plt.show()
            print("Predicted label: " + pred_label)
            print("Actual label: " + actual_label)

            print("*********\n")


class TIM_GD(TIM):
    @tim_ingredient.capture
    def __init__(self, lr, model):
        super().__init__(model=model)
        self.lr = lr

    def run_adaptation(self, support, query, y_s, y_q, shot, model_name, callback):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
            callback : VisdomLogger or None to plot in live our metrics


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()
        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()

        plot_title_testacc = str(shot) + "-shot_testacc/" + str(model_name)
        plot_title_mutualinfo = str(shot) + "-shot_mutualinfo/" + str(model_name)
        plot_title_loss = str(shot) + "-shot_testloss/" + str(model_name)

        for i in tqdm(range(self.iter)):
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)
                    ).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)
                            ).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) *
                       torch.log(q_probs.mean(1))).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - \
                (self.loss_weights[1] * q_ent -
                 self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            self.model.eval()
            if callback is not None:
                P_q = self.get_logits(query).softmax(2).detach()
                prec = (P_q.argmax(2) == y_q).float().mean()
                callback.scalar('prec', i, prec, title='Precision')

            # tensorboard plot test_acc
            P_q = self.get_logits(query).softmax(2).detach()
            prec = (P_q.argmax(2) == y_q).float().mean()
            writer.add_scalar(plot_title_testacc, prec, i)

            # tensorboard plot mutual_info
            mutual_info = get_mi(probs=q_probs).float().mean()
            writer.add_scalar(plot_title_mutualinfo, mutual_info, i)

            # tensorboard plot loss
            writer.add_scalar(plot_title_loss, loss, i)

            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
            t0 = time.time()

        P_q = self.get_logits(query).softmax(2).detach()
        self.pred_vs_actual_indexDifference = (P_q.argmax(2) == y_q)
        self.predicted_label = P_q.argmax(2)
        self.actual_label = y_q


class TIM_ADM(TIM):
    @tim_ingredient.capture
    def __init__(self, model, alpha):
        super().__init__(model=model)
        self.alpha = alpha

    def q_update(self, P):
        """
        inputs:
            P : torch.tensor of shape [n_tasks, q_shot, num_class]
                where P[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        """
        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0  # Corresponds to the weight of the KL penalty
        alpha = l2 / l3
        beta = l1 / (l1 + l3)

        Q = (P ** (1+alpha)) / ((P ** (1+alpha)).sum(dim=1, keepdim=True)) ** beta
        self.Q = (Q / Q.sum(dim=2, keepdim=True)).float()

    def weights_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, num_classes]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        P_s = self.get_logits(support).softmax(2)
        P_q = self.get_logits(query).softmax(2)
        src_part = self.loss_weights[0] / (1 + self.loss_weights[2]) * \
            y_s_one_hot.transpose(1, 2).matmul(support)
        src_part += self.loss_weights[0] / (1 + self.loss_weights[2]) * (self.weights * P_s.sum(1, keepdim=True).transpose(1, 2)
                                                                         - P_s.transpose(1, 2).matmul(support))
        src_norm = self.loss_weights[0] / (1 + self.loss_weights[2]) * \
            y_s_one_hot.sum(1).view(n_tasks, -1, 1)

        qry_part = self.N_s / self.N_q * self.Q.transpose(1, 2).matmul(query)
        qry_part += self.N_s / self.N_q * (self.weights * P_q.sum(1, keepdim=True).transpose(1, 2)
                                           - P_q.transpose(1, 2).matmul(query))
        qry_norm = self.N_s / self.N_q * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def run_adaptation(self, support, query, y_s, y_q, shot, callback):
        """
        Corresponds to the TIM-ADM inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
            callback : VisdomLogger or None to plot in live our metrics


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()
        y_s_one_hot = get_one_hot(y_s)
        for i in tqdm(range(self.iter)):
            P_q = self.get_logits(query).softmax(2)
            self.q_update(P=P_q)
            self.weights_update(support, query, y_s_one_hot)
            t1 = time.time()
            if callback is not None:
                callback.scalar(
                    'acc', i, self.test_acc[-1].mean(), title='Accuracy')
                callback.scalars(['cond_ent', 'marg_ent'], i, [self.cond_entropy[-1].mean(),
                                                               self.entropy[-1].mean()], title='Entropies')
            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            t0 = time.time()
