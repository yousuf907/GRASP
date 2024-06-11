import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import shutil
import os
#import faiss
import math
import pickle
import copy
import utils_pq_LT as utils
from retrieve_any_layer import ModelWrapper
from scipy.stats import mode
from collections import defaultdict
from SLDA_Model3 import StreamingLDA
sys.setrecursionlimit(10000)
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.

def randint(max_val, num_samples):
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break
        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()

class MNModel(object):
    def __init__(self, num_classes, classifier_G='MobNetClassifyAfterLayer8',
            extract_features_from='model.features.7', classifier_F='MobNet_StartAt_Layer8', classifier_ckpt=None, 
            weight_decay=1e-5, lr_mode=None, lr_step_size=100, start_lr=0.1, end_lr=0.001, lr_gamma=0.5, num_samples=50, 
            mixup_alpha=0.1, grad_clip=None, num_channels=80, num_feats=14, 
            num_codebooks=8, codebook_size=256, max_buffer_size=None, 
            sleep_batch_size=128, sleep_epoch=50, sleep_lr=0.2, seed=1):
        ### make the classifier
        self.classifier_F = utils.build_classifier(classifier_F, classifier_ckpt, num_classes=num_classes)
        core_model = utils.build_classifier(classifier_G, classifier_ckpt, num_classes=None)
        self.classifier_G = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

        ## OPTIMIZER
        params = self.get_layerwise_params(self.classifier_F, sleep_lr) # >> better
        self.optimizer = optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.lr_mode = lr_mode
        self.lr_step_size = lr_step_size
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_gamma = lr_gamma
        self.num_classes = num_classes  # 1000
        self.num_samples = num_samples 
        self.num_channels = num_channels
        self.num_feats = num_feats
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_buffer_size = max_buffer_size
        self.sleep_lr=sleep_lr
        self.sleep_bs = sleep_batch_size
        self.sleep_epoch = sleep_epoch
        ## SLDA
        #self.slda = StreamingLDA(960, num_classes)
        self.slda = StreamingLDA(1280, num_classes, shrinkage_param=1e-2) # >> optimums
        _set_seed(seed)

    #### ///// LayerWiseLR ///// ####
    def get_layerwise_params(self, classifier, lr):
        trainable_params = []
        layer_names = []
        lr_mult = 0.99
        for idx, (name, param) in enumerate(classifier.named_parameters()):
            layer_names.append(name)
        # reverse layers
        layer_names.reverse()
        # store params & learning rates
        for idx, name in enumerate(layer_names):
            # append layer parameters
            trainable_params += [{'params': [p for n, p in classifier.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]
            # update learning rate
            lr *= lr_mult
        return trainable_params
    
    def get_trainable_params(self, classifier, start_lr):
        trainable_params = []
        for k, v in classifier.named_parameters():
            trainable_params.append({'params': v, 'lr': start_lr})
        return trainable_params

    def get_penultimate_feat(self, codes, opq, pq): # codes : 1 x 14 x 14 x 8
        model_clone = copy.deepcopy(self.classifier_F)
        model_clone.cuda()
        model_clone.eval()
        codes = np.reshape(codes, (self.num_feats * self.num_feats, self.num_codebooks)) # 196 x 8                    
        recon = pq.decode(codes) #pq # 196 x 80
        recon = opq.reverse_transform(recon) #opq # 196 x 80
        recon = np.reshape(recon, (-1, self.num_feats, self.num_feats, self.num_channels))  # 1x14x14x80
        recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # 1x80x14x14
        feature = model_clone.get_penultimate_feature(recon) # 1x1280
        feature = F.normalize(feature, p=2.0, dim=1) # 1 x 1280
        return feature.squeeze()
    

    ##### ------------------------- #####
    ##### ----- UPDATE BUFFER ----- #####
    ##### ------------------------- #####
    def update_buffer(self, opq, pq, curr_loader, latent_dict, rehearsal_ixs, 
                 class_id_to_item_ix_dict, counter):
        start_time = time.time()
        classifier_G = self.classifier_G.cuda()
        classifier_G.eval()
        feat_ext = copy.deepcopy(self.classifier_F)
        feat_ext.cuda()
        feat_ext.eval()
        start_ix=0
        recent_lbls = np.zeros((len(curr_loader.dataset)))
        ### ------ New Samples ------ ###
        for batch_images, batch_labels, batch_item_ixs in curr_loader: # New classes
            end_ix = start_ix + batch_labels.shape[0]
            recent_lbls[start_ix:end_ix] = batch_labels.squeeze()
            start_ix = end_ix
            # get features from G and latent codes from PQ
            data_batch = classifier_G(batch_images.cuda()).cpu().numpy() # N x 80 x 14 x 14
            data_batch = np.transpose(data_batch, (0, 2, 3, 1)) # N x 14 x 14 x 80
            data_batch = np.reshape(data_batch, (-1, self.num_channels)) # 196N x 80
            data_batch = opq.apply_py(np.ascontiguousarray(data_batch, dtype=np.float32)) #opq # 196N x 80
            codes = pq.compute_codes(data_batch) # 196N x 8
            codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.num_codebooks)) # Nx14x14x8
            # put codes and labels into buffer (dictionary)
            for x, y, item_ix in zip(codes, batch_labels, batch_item_ixs): # x dim: 1x7x7x32
                # Add new data point to dict (new class)
                latent_dict[int(item_ix.numpy())] = [x, y.numpy()]
                rehearsal_ixs.append(int(item_ix.numpy()))
                class_id_to_item_ix_dict[int(y.numpy())].append(int(item_ix.numpy()))
                # if buffer is full, randomly replace previous example from class with most samples
                if self.max_buffer_size is not None and counter.count >= self.max_buffer_size:
                    # class with most samples and random item_ix from it
                    max_key = max(class_id_to_item_ix_dict, key=lambda x: len(class_id_to_item_ix_dict[x]))
                    max_class_list = class_id_to_item_ix_dict[max_key]
                    rand_item_ix = random.choice(max_class_list)
                    # remove the random_item_ix from all buffer references
                    max_class_list.remove(rand_item_ix)
                    latent_dict.pop(rand_item_ix)
                    rehearsal_ixs.remove(rand_item_ix)
                else:
                    counter.update()
            ## fit SLDA to new data
            new_data = self.decode(codes, feat_ext, opq, pq)
            if len(batch_labels) > 1:
                new_labels = batch_labels.squeeze()
            else:
                new_labels = batch_labels
            for x, y in zip(new_data, new_labels):
                self.slda.fit(x, y.view(1, ), sleep=False) # False for not updating Cov matrix
        spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("Time spent in buffer update process (in mins):", spent_time)
        recent_class_list = np.unique(recent_lbls)
        return latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list


    def decode(self, codes_batch, classifier_F, opq, pq):
        codes = np.reshape(codes_batch, (
                codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # N*14*14 x 8
        data_batch_recon = pq.decode(codes) #pq # 196N x 80
        data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
        data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # N x 14 x 14 x 80
        data_batch_recon = torch.from_numpy(
            np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # N x 80 x 14 x 14
        new_data = classifier_F.get_penultimate_feature(data_batch_recon) # N x 1280 ## pre-activation
        return new_data


    def dist_to_centroid(self, opq, pq, latent_dict, rehearsal_ixs):
        start_time = time.time()
        dist_dict = {}
        model_clone = copy.deepcopy(self.classifier_F)
        model_clone.cuda()
        model_clone.eval()
        codes = np.empty((len(rehearsal_ixs), self.num_feats, self.num_feats, 
            self.num_codebooks), dtype=np.uint8) # Nx7x7x32
        lbl = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        ixs = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        for ii, v in enumerate(rehearsal_ixs):
            codes[ii] = latent_dict[v][0]
            lbl[ii] = torch.from_numpy(latent_dict[v][1])
            ixs[ii] = v
        class_list = np.unique(lbl)
        for c in class_list:
            class_codes = codes[lbl == c]  # filter codes by class c # NC x 14 x 14 x 8
            class_ixs = ixs[lbl == c]
            class_codes = np.reshape(class_codes, (
                class_codes.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 196NC x 8                    
            recon = pq.decode(class_codes) # PQ # 196NC x 80
            recon = opq.reverse_transform(recon) #opq # 196NC x 80
            recon = np.reshape(recon, (-1, self.num_feats, self.num_feats, self.num_channels))  # NCx14x14x80
            recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # NCx80x14x14

            with torch.no_grad():
                features = model_clone.get_penultimate_feature(recon) # NCx1280

            mean_feature = torch.mean(features, 0) # mean across samples i.e., row wise NCx1280 -> 1280

            ## L2 Distance -- Square Root
            features = features.detach().cpu().numpy()
            mean_feature = mean_feature.detach().cpu().numpy()
            mean_feature = np.reshape(mean_feature, (-1, 1280))

            #dist = euclidean_distances(mean_feature, features) # 1 x NC
            dist = cosine_distances(mean_feature, features) # 1 x NC
            dist = dist.squeeze()
            class_ixs = np.array(class_ixs, dtype=np.int32) # important for speed up!
            dist_dict[c] = [dist, class_ixs, dist]
            ts = int((time.time() - start_time)/60) # in mins
        print("Elasped time in distance computation is:", ts)
        return dist_dict, class_list

    def update_new_nodes(self, class_list):
        start_time = time.time()
        bias=torch.ones(1).cuda()
        for k in class_list:
            k = torch.tensor(k, dtype=torch.int32)
            mu_k = self.slda.grab_mean(k)
            self.classifier_F.state_dict()['model.classifier.3.weight'][k].copy_(mu_k)
            self.classifier_F.state_dict()['model.classifier.3.bias'][k] = bias
        print('Elapsed Time for New Weight Init (in SEC): %0.3f' % (time.time() - start_time))

    def grasp_sampling(self, dist_dict, class_list, num_iters):
        rehearsal_idxs = []
        count = 0
        budget = self.sleep_bs*num_iters
   
        while count < budget:
            for i in range(len(class_list)):
                c = class_list[i] # class
                dist_current_class = dist_dict[c][0]
                ixs_current_class = dist_dict[c][1]
                probas = np.array(dist_current_class) # > hard examples
                probas = 1 / (probas + 1e-10) # min distances get highest scores/ priorities > easy examples >> optimum
                p_curr_class = probas / np.linalg.norm(probas, ord=1)  # sum to 1
                sel_idx = np.random.choice(ixs_current_class, size=1, replace=False, p=p_curr_class)
                count += 1
                ###
                h = int(np.where(ixs_current_class==sel_idx)[0])
                dist_dict[c][0][h] += np.max(dist_dict[c][0])
                ###
                rehearsal_idxs.append(torch.from_numpy(sel_idx))
                if count >= budget:
                    break
        rehearsal_idxs = torch.cat(rehearsal_idxs, dim=0)
        rehearsal_idxs = torch.tensor(rehearsal_idxs[:budget], dtype=torch.int32)
        print("Num of selected samples:", len(rehearsal_idxs))
        assert len(rehearsal_idxs) <= budget
        rehearsal_idxs = np.array(rehearsal_idxs, dtype=np.int32)
        return rehearsal_idxs
        

    def encode_decode(self, batch_images, classifier_G, classifier_F, opq, pq):
        data_batch = classifier_G(batch_images.cuda()).cpu().numpy()  # N x 80 x 14 x 14
        data_batch = np.transpose(data_batch, (0, 2, 3, 1))  # N x 14 x 14 x 80
        data_batch = np.reshape(data_batch, (-1, self.num_channels))  # 196N x 80
        data_batch = opq.apply_py(np.ascontiguousarray(data_batch, dtype=np.float32)) #opq # 196N x 80
        codes = pq.compute_codes(np.ascontiguousarray(data_batch, dtype=np.float32)) #pq # 196N x 8
        data_batch_recon = pq.decode(codes) #pq # 196N x 80
        data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
        data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # N x 14 x 14 x 80
        data_batch_recon = torch.from_numpy(
            np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # N x 80 x 14 x 14
        #new_data = classifier_F.get_feature(data_batch_recon) # N x 960
        new_data = classifier_F.get_penultimate_feature(data_batch_recon) # N x 1280
        #new_data = F.normalize(new_data, p=2.0, dim=1) # L2 Norm >> sub-optimal
        return new_data

    ### ----------------------------------- ###
    ### -------- GRASP Rehearsal --------- ###
    ### ---------------------------------- ###
    def rehearsal_grasp(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr) # start_lr=0.2 optimum
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        total_stored_samples = len(rehearsal_ixs)
        print("\nTotal Number of Stored Samples:", total_stored_samples)
        print("Not using any augmentations..")
        bs = self.sleep_bs
        num_iter = num_iters
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5) # wd=1e-5 optimum
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
               max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1) ## OneCycle
        sampling_start_time = time.time()
        ### GRASP ###
        dist_dict, class_list = self.dist_to_centroid(opq, pq, latent_dict, rehearsal_ixs)
        to_be_replayed = self.grasp_sampling(dist_dict, class_list, num_iters)
        sampling_time = time.time() - sampling_start_time
        total_sampling_time = int(sampling_time / 60)  # in minutes
        print("\nTotal Sampling Time (in mins):", total_sampling_time)
        total_num_samples = len(to_be_replayed)
        print("Total Number of Replay Samples:", total_num_samples)
        ## Gather data for partial replay
        codes = np.empty((len(to_be_replayed), self.num_feats, 
                self.num_feats, self.num_codebooks), dtype=np.uint8)
        labels = torch.empty(len(to_be_replayed), dtype=torch.long).cuda()
        for ii, v in enumerate(to_be_replayed):
            v = v.item()
            codes[ii] = latent_dict[v][0]
            labels[ii] = torch.from_numpy(latent_dict[v][1])

        ### /// TRAINING STEP /// ###
        for i in range(num_iter):
            start = i*bs
            if start > (total_num_samples - bs):
                end = total_num_samples
            else:
                end = (i+1) * bs
            codes_batch = codes[start:end]
            labels_batch = labels[start:end].cuda()
            codes_batch = np.reshape(codes_batch, (
                codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 64*14*14 x 8
            data_batch_recon = pq.decode(codes_batch) #pq # 64*14*14 x 80
            data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
            data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # 128 x 14 x 14 x 80
            data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # 64 x 80 x 14 x 14
            ### fit on replay mini-batch
            output = classifier_F(data_batch_recon)  # 64 x 80 x 14 x 14
            loss = criterion(output, labels_batch)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            ### Update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())

            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time)


    ### --------------------------------------------- ###
    ### -------- Uniform Balanced Rehearsal --------- ###
    ### --------------------------------------------- ###
    def rehearsal_uniform_bal(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr) # start_lr=0.2 optimum
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        total_stored_samples = len(rehearsal_ixs)
        print("\nTotal Number of Stored Samples:", total_stored_samples)
        print("Not using any augmentations..")
        bs = self.sleep_bs
        num_iter = num_iters
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5) # wd=1e-5 optimum
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
               max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1) ## OneCycle
        sampling_start_time = time.time()
        ## Balanced Uniform ##
        to_be_replayed = self.uniform_balanced(latent_dict, rehearsal_ixs, num_iter)
        sampling_time = time.time() - sampling_start_time
        total_sampling_time = int(sampling_time / 60)  # in minutes
        print("\nTotal Sampling Time (in mins):", total_sampling_time)
        total_num_samples = len(to_be_replayed)
        print("Total Number of Replay Samples:", total_num_samples)
        ## Gather data for partial replay
        codes = np.empty((len(to_be_replayed), self.num_feats, 
                self.num_feats, self.num_codebooks), dtype=np.uint8)
        labels = torch.empty(len(to_be_replayed), dtype=torch.long).cuda()
        for ii, v in enumerate(to_be_replayed):
            v = v.item()
            codes[ii] = latent_dict[v][0]
            labels[ii] = torch.from_numpy(latent_dict[v][1])

        ### /// TRAINING STEP /// ###
        for i in range(num_iter):
            start = i*bs
            if start > (total_num_samples - bs):
                end = total_num_samples
            else:
                end = (i+1) * bs
            codes_batch = codes[start:end]
            labels_batch = labels[start:end].cuda()
            codes_batch = np.reshape(codes_batch, (
                codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 64*14*14 x 8
            data_batch_recon = pq.decode(codes_batch) #pq # 64*14*14 x 80
            data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
            data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # 128 x 14 x 14 x 80
            data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # 64 x 80 x 14 x 14
            ### fit on replay mini-batch
            output = classifier_F(data_batch_recon)  # 64 x 80 x 14 x 14
            loss = criterion(output, labels_batch)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            ### Update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())

            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time)


    def uniform_balanced(self, latent_dict, rehearsal_ixs, num_iters):
        ixs = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        lbl = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        for ii, v in enumerate(rehearsal_ixs):
            lbl[ii] = torch.from_numpy(latent_dict[v][1])
            ixs[ii] = v
        class_list = np.unique(lbl)
        replay_idxs = []
        k=1
        count = 0
        budget = self.sleep_bs*num_iters #64*num_iters
        while count < budget:
            for c in class_list:
                ixs_current_class = ixs[lbl == c]
                sel_idx = np.random.choice(ixs_current_class, size=k, replace=False)
                count += k
                replay_idxs.append(torch.from_numpy(sel_idx))
                if count >= budget:
                    break
        replay_idxs = torch.cat(replay_idxs, dim=0)
        replay_idxs = torch.tensor(replay_idxs[:budget], dtype=torch.int32)
        print("Num of selected samples:", len(replay_idxs))
        assert len(replay_idxs) <= budget
        replay_idxs = np.array(replay_idxs, dtype=np.int32)
        return replay_idxs


    # ------------------------------------------------------------------- #
    # ------ Joint Train on Base Data using feature augmentations ------- #
    # ------------------------------------------------------------------- #
    def joint_train_base(self, opq, pq, latent_dict, rehearsal_ixs, test_loader, save_dir, ckpt_file):
        print("Offline training top layers using feature augmentations..")
        best_acc1=0
        best_acc5=0
        start_time = time.time()
        classifier_F = nn.DataParallel(self.classifier_F).cuda()
        classifier_F.train()
        criterion1 = nn.CrossEntropyLoss(reduction='none').cuda()
        criterion2 = nn.CrossEntropyLoss().cuda()
        random_resize_crop = utils.RandomResizeCrop(self.num_feats, scale=(2 / 7, 1.0))
        participation_rate=0.6
        beta=1.0
        bs = 2*self.sleep_bs #128 #self.sleep_bs
        num_epochs = self.sleep_epoch
        num_iter = math.ceil(len(rehearsal_ixs) / bs)
        print('Number of Training Samples:', len(rehearsal_ixs))
        loss_arr = np.zeros(num_iter, np.float32)
        for epoch in range(num_epochs):
            X = np.empty(
                (len(rehearsal_ixs), self.num_feats, self.num_feats, self.num_codebooks),
                dtype=np.uint8) # Nx14x14x8
            y = torch.empty(len(rehearsal_ixs), dtype=torch.long).cuda()
            ixs = randint(len(rehearsal_ixs), len(rehearsal_ixs))
            ixs = [rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
            for ii, v in enumerate(ixs):
                X[ii] = latent_dict[v][0]
                y[ii] = torch.from_numpy(latent_dict[v][1])
            for i in range(num_iter):
                start = i * bs
                if start > (y.shape[0] - bs):
                    start = (y.shape[0] - bs)
                    end = y.shape[0]
                else:
                    end = (i+1) * bs
                codes_batch = X[start:end]
                labels_batch = y[start:end].cuda()
                # Reconstruct/decode samples with PQ
                codes_batch = np.reshape(codes_batch, (
                    codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 196N x 8                    
                data_batch_recon = pq.decode(codes_batch) #pq # 196N x 80
                data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
                data_batch_recon = np.reshape(data_batch_recon,
                        (-1, self.num_feats, self.num_feats, self.num_channels))  # 2*mb x 14 x 14 x 80
                data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # 2*mb x 80 x 14 x 14
                ### Perform random resize crop augmentation on each tensor
                transform_data_batch = torch.empty_like(data_batch_recon)
                for tens_ix, tens in enumerate(data_batch_recon):
                    transform_data_batch[tens_ix] = random_resize_crop(tens)
                data_batch_recon = transform_data_batch

                if np.random.rand(1) < participation_rate:
                    ### MIXUP: Do mixup between two batches of previous data
                    num_instances = math.ceil(data_batch_recon.shape[0]/2)
                    x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                        data_batch_recon[:num_instances], labels_batch[:num_instances],
                        data_batch_recon[num_instances:], labels_batch[num_instances:],
                        alpha=0.1)
                    data = torch.empty((num_instances, self.num_channels, self.num_feats, self.num_feats))  # mb x 80 x 14 x 14
                    data = x_prev_mixed.clone()  # mb x 80 x 14 x 14
                    labels_a = torch.zeros(num_instances).long()  # mb
                    labels_b = torch.zeros(num_instances).long()  # mb
                    labels_a = prev_labels_a
                    labels_b = prev_labels_b
                    ### fit on replay mini-batch plus new sample
                    output = classifier_F(data.cuda())  # mb x 80 x 14 x 14
                    ### Manifold MixUp
                    loss = self.mixup_criterion(criterion1, output, labels_a.cuda(), labels_b.cuda(), lam)
                    loss = loss.mean()
                else:
                    ##### /// CutMix /// #######
                    num_instances = math.ceil(data_batch_recon.shape[0]/2)
                    input_a = data_batch_recon[:num_instances] # first 64
                    input_b = data_batch_recon[num_instances:] # last 64
                    lam = np.random.beta(beta, beta)
                    target_a = labels_batch[:num_instances] # first 64
                    target_b = labels_batch[num_instances:] # last 64
                    bbx1, bby1, bbx2, bby2 = self.rand_bbox(input_a.size(), lam)
                    input_a[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_a.size()[-1] * input_a.size()[-2]))
                    # compute output
                    output = classifier_F(input_a)
                    loss = criterion2(output, target_a) * lam + criterion2(output, target_b) * (1. - lam)
                    ######## /// ########

                self.optimizer.zero_grad() ## zero out grads before backward pass because they are accumulated
                loss.backward()
                self.optimizer.step()
                loss_arr[i] = loss.item()
            self.lr_scheduler.step()
            
            if (epoch + 1) % 1 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                probas, true = self.predict(test_loader, opq, pq)
                top1, top5 = utils.accuracy(probas, true, topk=(1, 5))
                is_best = top1 > best_acc1
                best_acc1 = max(top1, best_acc1)
                best_acc5 = max(top5, best_acc5)
                print("Epoch:", (epoch + 1), "--Loss: %1.5f" % np.mean(loss_arr),
                 "--Val_acc1: %1.5f" % top1, "--Best_acc1: %1.5f" % best_acc1, "--Best_acc5: %1.5f" % best_acc5)
                self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': classifier_F.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                }, is_best, save_dir, ckpt_file)
        print('\nBest Top1 Accuracy (%):', best_acc1)
        print('Time Spent in base initialization (in mins): %0.3f' % ((time.time() - start_time)/60))
    

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
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

    
    def mixup_data(self, x1, y1, x2, y2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x1 + (1 - lam) * x2
        y_a, y_b = y1, y2
        return mixed_x, y_a, y_b, lam

    # Mix Up Criterion #
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())

    def save_checkpoint(self, state, is_best, save_dir, ckpt_file):
        torch.save(state, os.path.join(save_dir, ckpt_file))
        if is_best:
            shutil.copyfile(os.path.join(save_dir, ckpt_file), os.path.join(save_dir, 'best_' + ckpt_file))
    
    def save(self, seed, save_full_path): #, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, opq, pq):
        if not os.path.exists(save_full_path):
            os.makedirs(save_full_path)
        state = {
            'model_state_dict': self.classifier_F.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print(f'\nSaving DNN model to {save_full_path}')
        torch.save(state, os.path.join(save_full_path, 'grasp_classifier_seed_%d.pth' % seed))
        
        ## get OPQ parameters
        #d=self.num_channels
        #A = faiss.vector_to_array(opq.A).reshape(d, d) ## from Swig Object to np array
        #b = faiss.vector_to_array(opq.b) ## from Swig Object to np array

        ## get PQ centroids/codebooks
        #centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        ## save in a dictionary
        #d = {'latent_dict': latent_dict, 'rehearsal_ixs': rehearsal_ixs,
        #     'class_id_to_item_ix_dict': class_id_to_item_ix_dict, 
        #     'opq_A': A, 'opq_b': b, 'pq_centroids': centroids}
        #with open(os.path.join(save_full_path, 'buffer_%d.pkl' % inc), 'wb') as f:
        #    pickle.dump(d, f)

    def resume(self, inc, resume_full_path):
        inc=100
        #print(f'\nResuming DNN model from {resume_full_path}')
        #state = torch.load(os.path.join('./' + resume_full_path, 'best_' + resume_full_path + '.pth'))
        #utils.safe_load_dict(self.classifier_F, state['state_dict'], should_resume_all_params=False)
        ## sanity check whether two checkpoints are similar ##
        #old_state=state['state_dict']
        #new_state = self.classifier_F.state_dict()
        #for k in old_state: # pretrained checkpoint
        #    assert torch.equal(old_state[k].cpu(), new_state[k[len("module."):]]), k
        #print("Successfully performed sanity check!!")
        # load parameters
        with open(os.path.join(resume_full_path, 'buffer_%d.pkl' % inc), 'rb') as f:
            d = pickle.load(f)
        nbits = int(np.log2(self.codebook_size))
        pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)
        opq = faiss.OPQMatrix(self.num_channels, self.num_codebooks)
        opq.pq = pq
        faiss.copy_array_to_vector(d['pq_centroids'].ravel(), pq.centroids)
        faiss.copy_array_to_vector(d['opq_A'].ravel(), opq.A)
        faiss.copy_array_to_vector(d['opq_b'].ravel(), opq.b)
        opq.is_trained = True
        opq.is_orthonormal = True
        return d['latent_dict'], d['rehearsal_ixs'], d['class_id_to_item_ix_dict'], opq, pq

    def resume2(self, seed):
        #resume_full_path = './places_LT_rehearsal_uniform_class_balanced/grasp_classifier_seed_' + str(seed) +  '.pth'
        resume_full_path = './Places_LT_exps_limited_memory_20K/grasp_classifier_seed_' + str(seed) +  '.pth'
        print(f'\nResuming DNN model from {resume_full_path}')
        state = torch.load(resume_full_path)
        utils.safe_load_dict(self.classifier_F, state['model_state_dict'], should_resume_all_params=False)
        ## sanity check whether two checkpoints are similar ##
        old_state=state['model_state_dict']
        new_state = self.classifier_F.state_dict()
        for k in old_state: # pretrained checkpoint
            assert torch.equal(old_state[k].cpu(), new_state[k].cpu()), k
        print("Successfully performed sanity check!!")

    ### --------------------------------- ###
    ### ---------- Inference ------------ ###
    ### --------------------------------- ###
    def predict(self, data_loader, opq, pq):
        with torch.no_grad():
            self.classifier_F.eval()
            self.classifier_F.cuda()
            self.classifier_G.eval()
            self.classifier_G.cuda()
            probas = torch.zeros((len(data_loader.dataset), self.num_classes), dtype=torch.float64)
            all_lbls = torch.zeros((len(data_loader.dataset)))
            #print("Number of samples in test set:", len(data_loader.dataset))
            start_ix = 0
            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()
                ## get G features
                data_batch = self.classifier_G(batch_x).cpu().numpy()  # N x 80 x 14 x 14
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))  # N x 14 x 14 x 80
                data_batch = np.reshape(data_batch, (-1, self.num_channels))  # 196N x 80
                data_batch = opq.apply_py(np.ascontiguousarray(data_batch, dtype=np.float32)) #opq # 196N x 80
                codes = pq.compute_codes(np.ascontiguousarray(data_batch, dtype=np.float32)) #pq # 196N x 8
                data_batch_reconstructed = pq.decode(codes) #pq # 196N x 80
                data_batch_reconstructed = opq.reverse_transform(data_batch_reconstructed) #opq # 196N x 80
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                            (-1, self.num_feats, self.num_feats, self.num_channels))  # N x 14 x 14 x 80
                data_batch_reconstructed = torch.from_numpy(
                    np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()  # N x 80 x 14 x 14
                batch_lbls = batch_lbls.cuda()
                logits = self.classifier_F(data_batch_reconstructed)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = F.softmax(logits.data, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix
        return probas.numpy(), all_lbls.int().numpy()

    ## end ##