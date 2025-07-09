import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint
from src.utils import load_pickle, save_pickle
from src.utils import shuffle_list
from src.datasets.ingredient import get_dataloader
import os
import torch
import collections
import torch.nn.functional as F
from src.tim import TIM, TIM_ADM, TIM_GD
from src.embedding_propagation import EmbeddingPropagation
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

embed_prop = EmbeddingPropagation()
reducer = umap.UMAP(random_state=42)
eval_ingredient = Ingredient('eval')


@eval_ingredient.config
def config():
    number_tasks = 10000
    n_ways = 5
    query_shots = 15
    method = 'baseline'
    model_tag = 'best'
    target_data_path = None  # Only for cross-domain scenario
    target_split_dir = None  # Only for cross-domain scenario
    plt_metrics = ['accs']
    shots = [1, 5]
    used_set = 'test'  # can also be val for hyperparameter tuning


class Evaluator:
    @eval_ingredient.capture
    def __init__(self, device, ex):
        self.device = device
        self.ex = ex

    @eval_ingredient.capture
    def run_full_evaluation(self, model, model_path, model_name, model_tag, shots, method, callback):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shots : Number of support shots to try

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        print("=> Runnning full evaluation with method: {}".format(method))

        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=model_tag)

        # Get loaders
        loaders_dic = self.get_loaders()

        # Extract features (just load them if already in memory)
        extracted_features_dic, filepath_allImages = self.extract_features(model=model,
                                                       model_path=model_path,
                                                       loaders_dic=loaders_dic)
        results = []
        for shot in shots:

            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,
                                        shot=shot)
            logs = self.run_task(task_dic=tasks,
                                 model=model,
                                 shot=shot,
                                 model_name=model_name,
                                 filepath_allImages=filepath_allImages,
                                 callback=callback)

            l2n_mean, l2n_conf = compute_confidence_interval(
                logs['acc'][:, -1])

            print('==> Meta Test: {} \nfeature\tL2N\n{}-shot \t{:.4f}({:.4f})'.format(
                  model_tag.upper(), shot, l2n_mean, l2n_conf))
            results.append(l2n_mean)

        return results

    def run_task(self, task_dic, model, model_name, shot, filepath_allImages, callback):

        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model=model)

        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        z_s, z_q = task_dic['z_s'], task_dic['z_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        oriLabelIndex_s, oriLabelIndex_q = task_dic['oriLabelIndex_s'], task_dic['oriLabelIndex_q']

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        oriLabelIndex_s = oriLabelIndex_s.long().squeeze(2).to(self.device)
        oriLabelIndex_q = oriLabelIndex_q.long().squeeze(2).to(self.device)

        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s)
        tim_builder.init_weights(
            support=support, y_s=y_s, query=query, y_q=y_q)
        
        # Visualize the features in 2D
        #for i in range(5):
        #    #mini_labels = get_dataset_labels('mini_imagenet')
        #    feat = query[i].clone().detach().cpu()
        #    lab = y_q[i].clone().detach().cpu()
        #    #lab_name = []
        #    #for index in oriLabelIndex_q[0].clone().detach():
        #    #  n = mini_labels[index]
        #    #  name = get_dataset_label_name('mini_imagenet', n)
        #    #  lab_name.append(name)
        #    reducer.fit(feat)
        #    embedding = reducer.transform(feat)
        #    plt.scatter(embedding[:, 0], embedding[:, 1], c=lab, cmap='tab10', s=5)
        #    plt.gca().set_aspect('equal', 'datalim')
        #    #plt.title('UMAP projection of Feature Vectors', fontsize=12);
        #    plt.show()

        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q,
                                  shot=shot, model_name=model_name, callback=callback)

        # Run visual checkings on query images
        #tim_builder.run_visual_analysis(img_index=x_q, oriLabelIndex=oriLabelIndex_q, raw_images_path=filepath_allImages, device=self.device)

        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    @eval_ingredient.capture
    def get_tim_builder(self, model, method):
        # Initialize TIM classifier builder
        tim_info = {'model': model}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError(
                "Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    @eval_ingredient.capture
    def get_loaders(self, used_set, target_data_path, target_split_dir):
        # First, get loaders
        loaders_dic = {}
        loader_info = {'aug': False, 'shuffle': False, 'out_name': False}

        if target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': target_data_path,
                                'split_dir': target_split_dir})

        train_loader = get_dataloader('train', **loader_info)
        loaders_dic['train_loader'] = train_loader

        test_loader = get_dataloader(used_set, **loader_info)
        loaders_dic.update({'test': test_loader})
        return loaders_dic

    @eval_ingredient.capture
    def extract_features(self, model, model_path, model_tag, used_set, loaders_dic):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
            filepath_allImages : file path to all images (based on DataLoader)
        """

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        filepath_allImages = os.path.join(save_dir, 'filepath_allImages.plk')

        if os.path.isfile(filepath) and os.path.isfile(filepath_allImages):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic, filepath_allImages     

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            all_images = []
            for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic['test'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs)
                outputs = embed_prop(outputs)  # ep
                outputs = embed_prop(outputs)  # ep
                all_features.append(outputs.cpu())
                all_labels.append(labels)
                all_images.append(inputs.cpu())

            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            all_images = torch.cat(all_images, 0)

            all_images_index = torch.arange(0, all_images.size(0), 1) # 0...N_of_Image indexes to track the images later

            extracted_features_dic = {'concat_features': all_features,
                                      'concat_labels': all_labels,
                                      'all_images_index': all_images_index
                                      }

        print(" ==> Saving features to {}".format(filepath))
        print(" ==> Saving all images tensor to {}".format(filepath_allImages))
        save_pickle(filepath, extracted_features_dic)
        save_pickle(filepath_allImages, all_images)

        return extracted_features_dic, filepath_allImages

    @eval_ingredient.capture
    def get_task(self, shot, n_ways, query_shots, extracted_features_dic):
        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features, labels and raw images
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
                                 x_support : Raw images Indexes of support set, torch.tensor of shape [n_ways * shot, 1]
                                 x_query : Raw images Indexes of query set, torch.tensor of shape [n_ways * query_shot, 1]
        """
        all_features = extracted_features_dic['concat_features']
        all_labels = extracted_features_dic['concat_labels']
        all_images_index = extracted_features_dic['all_images_index']
        all_classes = torch.unique(all_labels)

        samples_classes = np.random.choice(
            a=all_classes, size=n_ways, replace=False)

        support_samples = []
        query_samples = []

        support_imageIndex = []
        query_imageIndex = []

        support_oriLabels = []
        query_oriLabels = []

        for each_class in samples_classes:
            class_indexes = torch.where(all_labels == each_class)[0]
            indexes = np.random.choice(
                a=class_indexes, size=shot + query_shots, replace=False)

            support_samples.append(all_features[indexes[:shot]])
            query_samples.append(all_features[indexes[shot:]])
            support_imageIndex.append(all_images_index[indexes[:shot]])
            query_imageIndex.append(all_images_index[indexes[shot:]])

            for i in range(shot):
              support_oriLabels.append(each_class)

            for i in range(query_shots):
              query_oriLabels.append(each_class)

        y_support = torch.arange(n_ways)[:, None].repeat(1, shot).reshape(-1)
        y_query = torch.arange(n_ways)[:, None].repeat(1, query_shots).reshape(-1)
        support_oriLabels = torch.IntTensor(support_oriLabels)
        query_oriLabels = torch.IntTensor(query_oriLabels)

        z_support = torch.cat(support_samples, 0)
        z_query = torch.cat(query_samples, 0)

        x_support = torch.cat(support_imageIndex, 0)
        x_query = torch.cat(query_imageIndex, 0)

        # Feature Fusing
        x_support, y_support, z_support, support_oriLabels = shuffle_list(x_support, y_support, z_support, support_oriLabels)
        x_query, y_query, z_query, query_oriLabels = shuffle_list(x_query, y_query, z_query, query_oriLabels)

        task = {'z_s': z_support, 'y_s': y_support, 'x_s': x_support, 'oriLabelIndex_s': support_oriLabels,
                'z_q': z_query, 'y_q': y_query, 'x_q': x_query, 'oriLabelIndex_q': query_oriLabels}
        
        return task

    @eval_ingredient.capture
    def generate_tasks(self, extracted_features_dic, shot, number_tasks):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot]
                            x_support : torch.tensor of shape [number_tasks, n_ways * shot, 1]
                            x_query : torch.tensor of shape [number_tasks, n_ways * query_shot, 1] }
        """
        print(f" ==> Generating {number_tasks} tasks ...")
        tasks_dics = []
        for _ in warp_tqdm(range(number_tasks), False):
            task_dic = self.get_task(
                shot=shot, extracted_features_dic=extracted_features_dic)
            tasks_dics.append(task_dic)

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)

        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(
                n_tasks)], dim=0).view(n_tasks, n_samples, -1)

        return merged_tasks