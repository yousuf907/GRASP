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
import utils_mobvit as utils
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
    torch.backends.cudnn.benchmark = False
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
    def __init__(self, num_classes, classifier_G='MobViTClassifyAfterLayer4',
            extract_features_from='model.stages.3.0', classifier_F='MobViT_StartAt_Layer4', classifier_ckpt=None, 
            weight_decay=1e-2, lr_mode=None, lr_step_size=100, start_lr=0.1, end_lr=0.001, lr_gamma=0.5, num_samples=50, 
            sleep_samples=50000, mixup_alpha=0.1, grad_clip=None, num_channels=128, num_feats=16, 
            num_codebooks=8, codebook_size=256, max_buffer_size=None, 
            sleep_batch_size=128, sleep_epoch=50, sleep_lr=1e-4):
        # make the classifier
        self.classifier_F = utils.build_classifier(classifier_F, classifier_ckpt, num_classes=num_classes)
        core_model = utils.build_classifier(classifier_G, classifier_ckpt, num_classes=None)
        self.classifier_G = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

        ## Test output size of extracted features
        in_tensor = torch.ones((10, 3, 256, 256)).cuda()
        self.classifier_G.cuda()
        y1 = self.classifier_G(in_tensor)
        print('Shape of y1:', y1.shape) # 10 x 128 x 16 x 16
        assert y1.shape[0] == in_tensor.shape[0]
        assert y1.shape[1] == num_channels # 128
        assert y1.shape[2] == num_feats # 16
        assert y1.shape[3] == num_feats # 16
        print("Feature extraction sanity check passed!")

        # OPTIMIZER
        #params = self.get_trainable_params(self.classifier_F, start_lr)
        params = self.get_layerwise_params(self.classifier_F, sleep_lr)
        self.optimizer = optim.AdamW(params, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        self.lr_mode = lr_mode
        self.lr_step_size = lr_step_size
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_gamma = lr_gamma
        self.num_classes = num_classes  # 1000
        self.num_samples = num_samples
        self.sleep_samples = sleep_samples ####    
        self.num_channels = num_channels
        self.num_feats = num_feats
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_buffer_size = max_buffer_size
        self.sleep_lr=sleep_lr
        self.wd = weight_decay
        self.sleep_bs = sleep_batch_size
        self.sleep_epoch = sleep_epoch
        ## SLDA
        self.slda = StreamingLDA(640, num_classes, shrinkage_param=1e-2) # >> optimal
        _set_seed(1993)

    #### ///// LayerWiseLR ///// ####
    def get_layerwise_params(self, classifier, lr):
        trainable_params = []
        layer_names = []
        lr_mult = 0.99 #0.99
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
            new_labels = batch_labels.squeeze()
            for x, y in zip(new_data, new_labels):
                self.slda.fit(x, y.view(1, ), sleep=False) # False for not updating Cov matrix
            
        spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("Time spent in buffer update process (in mins):", spent_time)
        recent_class_list = np.unique(recent_lbls)
        return latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list

    def update_new_nodes(self, class_list):
        start_time = time.time()
        bias=torch.ones(1).cuda()
        for k in class_list:
            k = torch.tensor(k, dtype=torch.int32)
            mu_k = self.slda.grab_mean(k)
            self.classifier_F.state_dict()['model.fc.weight'][k].copy_(mu_k*0.01)
            self.classifier_F.state_dict()['model.fc.bias'][k] = bias*0.01


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

            num_data = len(class_ixs) # NC
            num_iter = math.ceil(num_data / 64)
            features = torch.zeros((num_data, 640), dtype=torch.float32) # NC x 1280
            bs=64
            with torch.no_grad():
                for i in range(num_iter):
                    start = i * bs
                    if (i+1) * bs > num_data:
                        end = num_data
                    else:
                        end = (i+1) * bs
                    recon_batch = recon[start:end]
                    features[start:end] = model_clone.get_penultimate_feature(recon_batch) # bsx640

            mean_feature = torch.mean(features, 0) # mean across samples i.e., row wise NCx640 -> 640
            features = features.detach().cpu().numpy()
            mean_feature = mean_feature.detach().cpu().numpy()
            mean_feature = np.reshape(mean_feature, (-1, 640))

            #dist = euclidean_distances(mean_feature, features) # 1 x NC
            dist = cosine_distances(mean_feature, features) # 1 x NC
            dist = dist.squeeze()
            #print('shape of dist:', dist.shape, 'for class:', c)
            class_ixs = np.array(class_ixs, dtype=np.int32) # important for speed up!
            dist_dict[c] = [dist, class_ixs]
            ts = int((time.time() - start_time)/60) # in mins
        print("Elasped time in distance computation is:", ts)
        return dist_dict, class_list


    def grasp_sampling(self, dist_dict, class_list, num_iters):
        rehearsal_idxs = []
        count = 0
        budget = self.sleep_bs*num_iters # 64*iters
        while count < budget:
            for i in range(len(class_list)):
                c = class_list[i] # class
                dist_current_class = dist_dict[c][0]
                ixs_current_class = dist_dict[c][1]
                probas = np.array(dist_current_class) # > hard examples
                probas = 1 / (probas + 1e-7) # min distances get highest scores/ priorities > easy examples
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
        assert len(rehearsal_idxs) <= budget
        rehearsal_idxs = np.array(rehearsal_idxs, dtype=np.int32)
        return rehearsal_idxs

    
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
        print("Num of samples:", len(replay_idxs))
        assert len(replay_idxs) <= budget
        replay_idxs = np.array(replay_idxs, dtype=np.int32)
        return replay_idxs

    
    ### Max Loss Replay ###
    def sample_max_loss(self, opq, pq, latent_dict, rehearsal_ixs, replay_loss=None, step=0):
        keep = self.sleep_bs # 64
        start_time = time.time()
        num_data = len(rehearsal_ixs)
        if step == 0:
            bs = 1000  # batch_size 256
            num_iter = math.ceil(num_data / bs)
            model_clone = self.classifier_F.cuda()
            model_clone.eval()
            outputs = torch.zeros((num_data, self.num_classes), dtype=torch.float32)  # logits: N x 1000
            criterion = nn.CrossEntropyLoss(reduction='none')
            codes = np.empty((len(rehearsal_ixs), self.num_feats, self.num_feats, 
                        self.num_codebooks), dtype=np.uint8) # Nx14x14x8
            labels = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
            for ii, v in enumerate(rehearsal_ixs):
                codes[ii] = latent_dict[v][0]
                labels[ii] = torch.from_numpy(latent_dict[v][1])

            with torch.no_grad():
                for i in range(num_iter):
                    start = i * bs
                    if start > (num_data - bs):
                        end = num_data
                    else:
                        end = (i+1) * bs
                    codes_batch = codes[start:end] # bsx14x14x8
                    # reconstruct/decode samples with PQ
                    codes_batch = np.reshape(codes_batch, (
                        codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # bs*14*14 x 8
                    data_batch_recon = pq.decode(codes_batch) #pq  # 196bs x 80
                    data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196bs x 80
                    data_batch_recon = np.reshape(data_batch_recon,
                                (-1, self.num_feats, self.num_feats, self.num_channels))  # bs x 14 x 14 x 80
                    data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # bs x 80 x 14 x 14
                    ## Obtain model's predictions/ logits
                    with torch.no_grad():
                        outputs[start:end] = model_clone(data_batch_recon)  # logits: mb x num_classes
                # compute replay probabilities
                losses = criterion(outputs, labels)  # N
                replay_loss = losses.cpu().numpy()

        ### compute sampling probability distribution
        replay_ixs = np.array(rehearsal_ixs, dtype=np.int32)
        proba = replay_loss # maximum loss has higher probability
        proba = proba + (1 - np.min(proba)) # min shift to 1
        p = proba / np.linalg.norm(proba, ord=1)  # sum to 1
        ##
        indices_all = np.arange(num_data)
        indices = list(np.random.choice(indices_all, size=keep, replace=False, p=p)) # probabilities
        kept_idx =  replay_ixs[indices]
        #kept_idx = list(np.random.choice(replay_ixs, size=keep, replace=False, p=p)) # probabilities
        kept_idx = np.array(kept_idx, dtype=np.int32)
        indices = np.array(indices, dtype=np.int32)
        assert len(kept_idx) == keep
        time_spent = time.time() - start_time # in seconds
        return kept_idx, indices, replay_loss, time_spent


    ### /// Distance Computation for Max Interference Replay /// ###
    def get_min_dist(self, xq, xb, yq, yb, Zb): # Query, xq : Nqx14x14x80 # Database, xb : Nbx14x14x80
        start_time = time.time()
        model_clone = copy.deepcopy(self.classifier_F)
        model_clone.cuda()
        model_clone.eval()
        batch = 5000 # hyperparam
        k = 1
        size = self.num_feats * self.num_feats  # 196 i.e., 14*14
        d = 1280 #960 #self.num_channels # feature dimension # 80
        nlist = 100
        ngpu = faiss.get_num_gpus() # num of available gpus
        print("number of gpu found:", ngpu)
        sub_q = self.num_codebooks  # 8
        nprobe = 10 # self.num_codebooks # 10
        nbits = int(np.log2(self.codebook_size)) # 8
        sample_size = xq.shape[0] # Nq
        num_iter = math.ceil(sample_size / batch)
        dc = torch.zeros((sample_size, self.num_classes), dtype=torch.float32) # Nq x 1000
        ic = torch.zeros((sample_size, self.num_classes), dtype=torch.long) # Nq x 1000

        with torch.no_grad():
            for c in torch.unique(yb):
                xc = xb[yb == c]  # filter codes by a class c # NCx14x14x80
                Zc = Zb[yb == c]
                ## Grab penultimate features
                xc = torch.from_numpy(np.transpose(xc, (0, 3, 1, 2))).cuda()  # NCx80x14x14
                #xc = model_clone.get_feature(xc) # Nx960
                xc = model_clone.get_penultimate_feature(xc) # N x 1280
                xc = F.normalize(xc, p=2.0, dim=1) # L2 norm # Nx1280
                xc = np.array(xc.detach().cpu(), dtype=np.float32)

                ### -----/// FAISS INDEX PART \\\----- ###
                res = [faiss.StandardGpuResources() for i in range(ngpu)]  # resources
                flat_config = []
                for i in range(ngpu):
                    #cfg = faiss.GpuIndexIVFPQConfig() # PQ
                    cfg = faiss.GpuIndexFlatConfig()
                    cfg.useFloat16 = True  # check here!!
                    cfg.interleavedLayout = True
                    cfg.device = i
                    flat_config.append(cfg)
                    res[i].noTempMemory()
                if ngpu == 1:
                    #index = faiss.GpuIndexIVFPQ(res[0], d, nlist, sub_q, nbits, faiss.METRIC_L2, flat_config[0])
                    index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
                else:
                    #indexes = [faiss.GpuIndexIVFPQ(res[i], d, nlist, sub_q, nbits, faiss.METRIC_L2, flat_config[i])
                    #            for i in range(ngpu)]
                    indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
                    index = faiss.IndexReplicas()
                    for sub_index in indexes:
                        index.addIndex(sub_index)
                index.reset()
                #index.train(xc) # not required for IndexFlatL2
                index.add(xc)
                #index.nprobe = nprobe # not required for IndexFlatL2
                for m in range(num_iter):
                    start = m * batch
                    if (m+1) * batch > sample_size:
                        end = sample_size
                    else:
                        end = (m+1) * batch
                    xq_small = xq[start:end] # batch x 14 x 14 x 80
                    xq_small = torch.from_numpy(np.transpose(xq_small, (0, 3, 1, 2))).cuda()  # batchx80x14x14
                    #xq_small = model_clone.get_feature(xq_small) # batchx960
                    xq_small = model_clone.get_penultimate_feature(xq_small) # bs x 1280
                    xq_small = F.normalize(xq_small, p=2.0, dim=1) # L2 norm
                    xq_small = np.array(xq_small.detach().cpu(), dtype=np.float32)
                    D, I = index.search(xq_small, k) # batch x k
                    D = D[:, -1]
                    I = I[:, -1]
                    D = np.sqrt(D) # L2 distance in square-root
                    dc[start:end, c] = torch.from_numpy(D.squeeze())
                    # indices
                    sel_ids0 = Zc[I.squeeze()] # 196*batch
                    ic[start:end, c] = sel_ids0
                del index
        # To avoid zeros in dc corresponding to absent/ unseen classes
        indices = np.where(dc == 0)
        dc[indices] = float('inf') # dim: Nq x 1000
        # To avoid min distance in dc corresponding to same class
        mask = np.arange(sample_size)
        #dc[mask, yb] = float('inf') # dim: Nq x 1000
        dc[mask, yq] = float('inf') # dim: Nq # hiding true labels since they've min distances!
        min_dist = dc.min(axis=1)[0] # dim: Nq
        pred_lbl = dc.argmin(axis=1)
        ic1 = ic[mask, pred_lbl].squeeze()
        print('\nTime Elapsed for L2 Dist Compute (in Mins): %0.3f' % ((time.time() - start_time)/60))
        return min_dist, ic1

    
    ## ------------------------------- ##
    ## /// SLDA New Class Sampling /// ##
    ## ------------------------------- ##
    def sample_slda_means(self, opq, pq, latent_dict, rehearsal_ixs, 
            new_class_list, mdist_dict=None, steps=0):
        start_time = time.time()
        sel_ixs_new = []
        sel_ixs_old = []
        old_rehearsal_ixs = []
        new_rehearsal_ixs = []
        keep=1
        bs=700
        if steps == 0:
            start_time0 = time.time()
            for ii, v in enumerate(rehearsal_ixs):
                if latent_dict[v][1] in new_class_list:
                    new_rehearsal_ixs.append(v)   
                else:
                    old_rehearsal_ixs.append(v)    
            old_rehearsal_ixs = np.array(old_rehearsal_ixs, dtype=np.int32)
            new_rehearsal_ixs = np.array(new_rehearsal_ixs, dtype=np.int32)
            ### new classes
            codes_new = np.empty((len(new_rehearsal_ixs), self.num_feats, self.num_feats, 
                            self.num_codebooks), dtype=np.uint8) # N x 14 x 14 x 8
            labels_new = torch.empty(len(new_rehearsal_ixs), dtype=torch.long) # N
            for ii, v in enumerate(new_rehearsal_ixs):
                codes_new[ii] = latent_dict[v][0] # NC x 14 x 14 x 8
                labels_new[ii] = torch.from_numpy(latent_dict[v][1]) # NC
            recon_new = np.reshape(codes_new, (
                codes_new.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # NC*14*14 x 8
            recon_new = pq.decode(recon_new) # NC*14*14 x 80
            recon_new = opq.reverse_transform(recon_new) #opq # 196NC x 80
            recon_new = np.reshape(recon_new, (-1, self.num_feats, self.num_feats, self.num_channels)) # NCx14x14x80
            ### old classes
            codes_old = np.empty((len(old_rehearsal_ixs), self.num_feats, self.num_feats, 
                            self.num_codebooks), dtype=np.uint8) # N x 14 x 14 x 8
            labels_old = torch.empty(len(old_rehearsal_ixs), dtype=torch.long) # N
            Z_old=[]
            for ii, v in enumerate(old_rehearsal_ixs):
                codes_old[ii] = latent_dict[v][0] # NC x 14 x 14 x 8
                labels_old[ii] = torch.from_numpy(latent_dict[v][1]) # NC
                Z_old.append(v)
            Z_old = torch.tensor(Z_old, dtype=torch.float32)
            recon_old = np.reshape(codes_old, (
                codes_old.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # NC*14*14 x 8
            recon_old = pq.decode(recon_old) # NC*14*14 x 80
            recon_old = opq.reverse_transform(recon_old) #opq # 196NC x 80
            recon_old = np.reshape(recon_old, (-1, self.num_feats, self.num_feats, self.num_channels)) # NCx14x14x80
            _, interf_idx = self.get_min_dist(recon_new, recon_old, labels_new, labels_old, Z_old)
            ####
            mdist_dict = {}
            model_clone = copy.deepcopy(self.classifier_F)
            model_clone.cuda()
            model_clone.eval()
            with torch.no_grad():
                for c in new_class_list:
                    c = int(c)
                    class_codes = codes_new[labels_new == c]  # Filter Codes by Class C # NC x 14 x 14 x 8
                    class_ixs = new_rehearsal_ixs[labels_new == c] # NC
                    class_interf = interf_idx[labels_new == c] # NC
                    recon = recon_new[labels_new == c] # NC x 14 x 14 x 80
                    recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # NCx80x14x14
                    #features = model_clone.get_penultimate_feature(recon) # N x 1280
                    with torch.no_grad():
                        features = model_clone.get_penultimate_feature(recon) # NCx1280
                    mean_feature = torch.mean(features, 0) # mean across samples i.e., row wise NCx1280 -> 1280
                    ## L2 Distance -- Square Root
                    features = features.detach().cpu().numpy()
                    mean_feature = mean_feature.detach().cpu().numpy()
                    mean_feature = np.reshape(mean_feature, (-1, 1280))

                    score = cosine_distances(mean_feature, features) # 1 x NC
                    score = score.squeeze()
                    class_ixs = np.array(class_ixs, dtype=np.int32) # important for speed up!

                    mdist_dict[c] = [score, class_ixs, class_interf]
                    #mdist_dict[c] = [mscores, class_ixs]
            ts0 = time.time() - start_time0 # in seconds
            print("Finished Cosine Distance Computation in SEC: %0.3f" % ts0)
        else:
            for c in new_class_list:
                scores = mdist_dict[c][0] # Cosine Distance Score/ distance > Min value is prioritized
                class_ixs = mdist_dict[c][1]
                ixs_old = mdist_dict[c][2]
                probas0 = np.array(scores)
                probas = 1 / (probas0 + 1e-7) # Min values get highest scores/ priorities
                p_center = probas / np.linalg.norm(probas, ord=1)  # sum to 1
                new_idx = np.random.choice(class_ixs, size=keep, replace=False, p=p_center)
                sel_ixs_new.append(new_idx)
                #p_center2 = probas0 / np.linalg.norm(probas0, ord=1)  # sum to 1
                #new_idx2 = np.random.choice(class_ixs, size=keep, replace=False, p=p_center2)
                #old_idx = ixs_old[class_ixs == new_idx2]
                old_idx = ixs_old[class_ixs == new_idx]
                sel_ixs_old.append(old_idx)
            sel_ixs_new = np.array(sel_ixs_new).squeeze()
            sel_ixs_old = np.array(sel_ixs_old).squeeze()
            assert len(sel_ixs_new) == len(new_class_list)
            assert len(sel_ixs_old) == len(new_class_list)
        ts = time.time() - start_time # in seconds
        return old_rehearsal_ixs, sel_ixs_new, sel_ixs_old, mdist_dict, ts

    
    def encode_decode(self, batch_images, classifier_G, classifier_F, opq, pq):
        data_batch = classifier_G(batch_images.cuda()).cpu().numpy()  # N x 80 x 14 x 14
        data_batch = np.transpose(data_batch, (0, 2, 3, 1))  # N x 14 x 14 x 80
        data_batch = np.reshape(data_batch, (-1, self.num_channels))  # 196N x 80
        data_batch = opq.apply_py(np.ascontiguousarray(data_batch, dtype=np.float32)) #opq # 196N x 80
        codes = pq.compute_codes(np.ascontiguousarray(data_batch, dtype=np.float32)) #pq # 196N x 8
        data_batch_recon = pq.decode(codes) #pq # 196N x 80
        data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196N x 80
        data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # Nx14x14x80
        data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # Nx80x14x14
        new_data = classifier_F.get_penultimate_feature(data_batch_recon) # N x 1280
        return new_data

    def decode(self, codes_batch, classifier_F, opq, pq):
        codes = np.reshape(codes_batch, (
                codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # N*16*16 x 8
        data_batch_recon = pq.decode(codes) #pq # 256N x 128
        data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 256N x 128
        data_batch_recon = np.reshape(data_batch_recon,
                    (-1, self.num_feats, self.num_feats, self.num_channels))  # N x 16 x 16 x 128
        data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # N x 128 x 16 x 16
        new_data = classifier_F.get_penultimate_feature(data_batch_recon) # N x 640
        return new_data
    
    
    ### Min Margin Sampling ###
    def sample_margin(self, opq, pq, latent_dict, rehearsal_ixs, labels=None, margins=None, step=0):
        keep = self.sleep_bs # 64
        start_time = time.time()
        num_data = len(rehearsal_ixs)
        if step == 0:
            bs = 1000  # batch_size 256
            num_iter = math.ceil(num_data / bs)
            model_clone = self.classifier_F.cuda()
            model_clone.eval()
            outputs = torch.zeros((num_data, self.num_classes), dtype=torch.float32)  # logits: N x 1000
            criterion = nn.CrossEntropyLoss(reduction='none')
            codes = np.empty((len(rehearsal_ixs), self.num_feats, self.num_feats, 
                        self.num_codebooks), dtype=np.uint8) # Nx14x14x8
            labels = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
            for ii, v in enumerate(rehearsal_ixs):
                codes[ii] = latent_dict[v][0]
                labels[ii] = torch.from_numpy(latent_dict[v][1])
            with torch.no_grad():
                for i in range(num_iter):
                    start = i * bs
                    if start > (num_data - bs):
                        end = num_data
                    else:
                        end = (i+1) * bs
                    codes_batch = codes[start:end] # bsx14x14x8
                    # reconstruct/decode samples with PQ
                    codes_batch = np.reshape(codes_batch, (
                        codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # bs*14*14 x 8
                    data_batch_recon = pq.decode(codes_batch) #pq  # 196bs x 80
                    data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 196bs x 80
                    data_batch_recon = np.reshape(data_batch_recon,
                                (-1, self.num_feats, self.num_feats, self.num_channels))  # bs x 14 x 14 x 80
                    data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # bs x 80 x 14 x 14
                    outputs[start:end] = model_clone(data_batch_recon)  # logits: mb x num_classes
                Fs = outputs.clone().detach()
                mask = torch.arange(num_data).cuda()
                Fs[mask, labels] = -float('inf')
                s_t = torch.argmax(Fs, dim=1)
                vals = outputs[mask, labels] - outputs[mask, s_t] # min values are prioritized
                margins = vals.cpu().numpy()
        # prob dist
        proba = margins
        proba = 1 / (proba + 1e-7) # min values get highest scores/ priorities
        proba = proba + (1 - np.min(proba)) # min shift to 1
        p = proba / np.linalg.norm(proba, ord=1)  # sum to 1
        replay_ixs = np.array(rehearsal_ixs, dtype=np.int32)
        ixs_all = np.arange(num_data)
        ixs_sel = list(np.random.choice(ixs_all, size=keep, replace=False, p=p)) # probabilities
        kept_idx =  replay_ixs[ixs_sel]
        kept_idx = np.array(kept_idx, dtype=np.int32)
        ixs_sel = np.array(ixs_sel, dtype=np.int32)
        assert len(kept_idx) == keep
        time_spent = time.time() - start_time
        return kept_idx, labels, margins, ixs_sel, time_spent


    ### ---------------------------------- ###
    ###  ///// Min Margin Rehearsal \\\\\ ###
    ### --------------------------------- ###
    def rehearsal_margin(self, opq, pq, latent_dict, rehearsal_ixs, num_iters): 
        start_time = time.time()
        total_loss = utils.CMA()
        self.classifier_F.requires_grad_(True)
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        #classifier_F = nn.DataParallel(self.classifier_F).cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        k = self.sleep_bs
        num_iter = num_iters
        mask = torch.arange(k).cuda()
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
            max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1)
        _, labels0, margins, _, ts = self.sample_margin(opq, pq, latent_dict, rehearsal_ixs)
        sampling_time = ts        
        for i in range(num_iter):
            chosen_rehearsal_ixs, labels0, margins, indices, ts = self.sample_margin(opq, pq, latent_dict, 
                    rehearsal_ixs, labels0, margins, i+1)
            sampling_time += ts #(ts1+ts2)
            codes = np.empty((k, self.num_feats, self.num_feats, self.num_codebooks), dtype=np.uint8) # 50x14x14x8
            labels = torch.empty(k, dtype=torch.long).cuda() # 32
            for ii, v in enumerate(chosen_rehearsal_ixs):
                codes[ii] = latent_dict[v][0]
                labels[ii] = torch.from_numpy(latent_dict[v][1])
            # Reconstruct
            codes = np.reshape(codes, (codes.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 100*14*14 x 8                 
            recon = pq.decode(codes) #pq #19600 x 80
            recon = opq.reverse_transform(recon) #opq # 19600 x 80
            recon = np.reshape(recon, (-1, self.num_feats, self.num_feats, self.num_channels))  # 100x14x14x80
            recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # 100 x 80 x 14 x 14
            ### fit on replay mini-batch
            classifier_F.train()
            output = classifier_F(recon)  # recon dim: 100 x 80 x 14 x 14, output dim: bs x 1000
            loss = criterion(output, labels)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            # update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())
            ## Update margins of mini-batch ##
            Fs = output.clone().detach()
            Fs[mask, labels] = -float('inf')
            s_t = torch.argmax(Fs, dim=1)
            vals = output[mask, labels] - output[mask, s_t] # min values are prioritized
            margins[indices] = vals.detach().cpu().numpy()
            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "--Loss: %1.5f" % total_loss.avg)
        print("\nSampling Time (in mins):", int(sampling_time/60))
        spent_time = int((time.time() - start_time) / 60) - int(sampling_time/60)  # in minutes
        print("Time Spent in Updating DNN (in mins):", spent_time)


    ### ------------------ ###
    ### /// Min Replay /// ###
    ### ------------------ ###
    def sample_min_replay(self, latent_dict, rehearsal_ixs, new_class_list, 
        replay_count=None, replay_ixs=None, step=0):
        start_time = time.time()
        keep = self.sleep_bs # 64
        if step == 0:
            new_count = 1
            old_count = 1
            replay_count = []
            replay_ixs = []
            for ii, v in enumerate(rehearsal_ixs):
                label = torch.from_numpy(latent_dict[v][1])
                replay_ixs.append(v)
                if label.item() in new_class_list:
                    replay_count.append(new_count)
                else:
                    replay_count.append(old_count)
        probas = np.array(replay_count)
        replay_ixs = np.array(replay_ixs)
        if np.min(probas) > 1:
            probas = probas + (1 - np.min(probas)) # min shift to 1
        probas = 1 / (probas + 1e-7) # min values get highest scores/ priorities
        p_min = probas / np.linalg.norm(probas, ord=1)  # sum to 1
        kept_idx = list(np.random.choice(replay_ixs, size=keep, replace=False, p=p_min))
        kept_idx = np.array(kept_idx, dtype=np.int32)
        assert len(kept_idx) == keep
        replay_count = np.array(replay_count)
        chosen_rehearsal_ixs = []
        for ii, ix in enumerate(kept_idx):
            chosen_rehearsal_ixs.append(ix)
            index1 = np.where(replay_ixs == ix)
            index1 = int(index1[0])
            replay_count[index1] = replay_count[index1] + 1
        time_spent = time.time() - start_time
        chosen_rehearsal_ixs = np.array(chosen_rehearsal_ixs, dtype=np.int32)
        return chosen_rehearsal_ixs, replay_count, replay_ixs, time_spent
    
    ### -------------------------- ###
    ###  ///// Min Rehearsal \\\\\ ###
    ### -------------------------- ###
    def rehearsal_minrep(self, opq, pq, latent_dict, rehearsal_ixs, new_class_list, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        self.classifier_F.requires_grad_(True)
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        #classifier_F = nn.DataParallel(self.classifier_F).cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        k = self.sleep_bs
        num_iter = num_iters # num_samples_per_sleep=9600*100=960K=0.96M
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
            max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1)
        _, replay_count, replay_ixs, ts = self.sample_min_replay(latent_dict, rehearsal_ixs, new_class_list)
        sampling_time = ts
        
        for i in range(num_iter):
            chosen_rehearsal_ixs, replay_count, replay_ixs, ts = self.sample_min_replay(latent_dict, rehearsal_ixs, 
                new_class_list, replay_count, replay_ixs, i+1)
            sampling_time += ts
            codes = np.empty((k, self.num_feats, self.num_feats, self.num_codebooks), dtype=np.uint8) # 50x14x14x8
            labels = torch.empty(k, dtype=torch.long).cuda() # 32
            for ii, v in enumerate(chosen_rehearsal_ixs):
                codes[ii] = latent_dict[v][0]
                labels[ii] = torch.from_numpy(latent_dict[v][1])
            # Reconstruct
            codes = np.reshape(codes, (codes.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 100*14*14 x 8                 
            recon = pq.decode(codes) #pq #19600 x 80
            recon = opq.reverse_transform(recon) #opq # 19600 x 80
            recon = np.reshape(recon, (-1, self.num_feats, self.num_feats, self.num_channels))  # 100x14x14x80
            recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # 100 x 80 x 14 x 14
            ### fit on replay mini-batch
            output = classifier_F(recon)  # 100 x 80 x 14 x 14
            loss = criterion(output, labels)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            # update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())
            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "--Loss: %1.5f" % total_loss.avg)
        print("Sampling Time (in mins):", int(sampling_time/60))
        spent_time = int((time.time() - start_time) / 60) - int(sampling_time/60)  # in minutes
        print("Time Spent in Updating DNN (in mins):", spent_time)


    ### ----------------------------------- ###
    ###  ///// Maximum Loss Rehearsal \\\\\ ###
    ### ---------------------------------- ###
    def rehearsal_ml(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        sleep_acc_all5=[]
        self.classifier_F.requires_grad_(True)
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        #classifier_F = nn.DataParallel(self.classifier_F).cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        ml_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        k = self.sleep_bs # 64
        num_iter = num_iters
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
            max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1)
        #vf = 500 # validation frequency
        chosen_rehearsal_ixs, _, replay_loss, ts = self.sample_max_loss(opq, pq, latent_dict, rehearsal_ixs)
        sampling_time = ts
        for i in range(num_iter):
            chosen_rehearsal_ixs, indices, replay_loss, ts = self.sample_max_loss(opq, pq, latent_dict, 
                    rehearsal_ixs, replay_loss, i+1)
            sampling_time += ts #(ts1+ts2)
            codes = np.empty((k, self.num_feats, self.num_feats, self.num_codebooks), dtype=np.uint8) # 50x14x14x8
            labels = torch.empty(k, dtype=torch.long).cuda() # 32
            for ii, v in enumerate(chosen_rehearsal_ixs):
                codes[ii] = latent_dict[v][0]
                labels[ii] = torch.from_numpy(latent_dict[v][1])
            # Reconstruct
            codes = np.reshape(codes, (codes.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 100*14*14 x 8                 
            recon = pq.decode(codes) #pq #19600 x 80
            recon = opq.reverse_transform(recon) #opq # 19600 x 80
            recon = np.reshape(recon, (-1, self.num_feats, self.num_feats, self.num_channels))  # 100x14x14x80
            recon = torch.from_numpy(np.transpose(recon, (0, 3, 1, 2))).cuda()  # 100 x 80 x 14 x 14
            ### fit on replay mini-batch
            output = classifier_F(recon)  # 100 x 80 x 14 x 14
            loss = criterion(output, labels)
            ml_loss = ml_criterion(output, labels)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            # update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())
            replay_loss[indices] = ml_loss.detach().cpu().numpy()
            
            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "--Loss: %1.5f" % total_loss.avg)
        print("Sampling Time (in mins):", int(sampling_time/60))
        spent_time = int((time.time() - start_time) / 60) - int(sampling_time/60)  # in minutes
        print("Time Spent in Updating DNN (in mins):", spent_time)

    
    ### ---------------------------------- ###
    ### -------- GRASP Rehearsal -------- ###
    ### --------------------------------- ###
    def rehearsal_grasp(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        #criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        total_stored_samples = len(rehearsal_ixs)
        print("Total Number of Stored Samples:", total_stored_samples)
        print("Not using any feature augmentations..")
        bs = self.sleep_bs # 64
        ### SAMPLING STEP ###
        num_iter = num_iters
        optimizer2 = optim.AdamW(params, weight_decay=self.wd)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
               max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1) ## OneCycle
        sampling_start_time = time.time()
        ### GRASP ###
        dist_dict, class_list = self.dist_to_centroid(opq, pq, latent_dict, rehearsal_ixs)
        to_be_replayed = self.grasp_sampling(dist_dict, class_list, num_iter)   
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
            if (i+1) % 500 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time) 

    ### -------------------------------------------- ###
    ### -------- Uniform Balanced Rehearsal -------- ###
    ### -------------------------------------------- ###
    def rehearsal_uniform_bal(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        #criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        total_stored_samples = len(rehearsal_ixs)
        print("Total Number of Stored Samples:", total_stored_samples)
        print("Not using any feature augmentations..")
        bs = self.sleep_bs # 64
        ### SAMPLING STEP ###
        num_iter = num_iters
        optimizer2 = optim.AdamW(params, weight_decay=self.wd)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
               max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1) ## OneCycle
        sampling_start_time = time.time()
        ## balanced uniform random sampling
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
            if (i+1) % 500 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time) 


    ### ------------------------------------ ###
    ### -------- Uniform Rehearsal -------- ###
    ### ----------------------------------- ###
    def rehearsal_uniform(self, opq, pq, latent_dict, rehearsal_ixs, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        #criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        total_stored_samples = len(rehearsal_ixs)
        print("Total Number of Stored Samples:", total_stored_samples)
        print("Not using any feature augmentations..")
        bs = self.sleep_bs # 64
        ### SAMPLING STEP ###
        num_iter = num_iters
        optimizer2 = optim.AdamW(params, weight_decay=self.wd)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
               max_lr=self.sleep_lr, steps_per_epoch=num_iter, epochs=1) ## OneCycle
        sampling_start_time = time.time()
        #### Uniform ##
        to_be_replayed = []
        rehearsal_ixs = np.array(rehearsal_ixs, dtype=np.int32) # important for speed up!
        for s in range(num_iter):   
            chosen_rehearsal_ixs = np.random.choice(rehearsal_ixs, size=bs, replace=False) # 64
            to_be_replayed.append(torch.from_numpy(chosen_rehearsal_ixs))
        to_be_replayed = torch.cat(to_be_replayed, dim=0)
        to_be_replayed = torch.tensor(to_be_replayed, dtype=torch.int32)  
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
            if (i+1) % 500 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time) 


    # ------------------------------------------------------------------- #
    # ------ Joint Train on Base Data using feature augmentations ------- #
    # ------------------------------------------------------------------- #
    def joint_train_base(self, opq, pq, latent_dict, rehearsal_ixs, 
        test_loader, save_dir, ckpt_file):
        print("Offline training top layers using CutMix feature augmentations..")
        best_acc1=0
        best_acc5=0
        start_time = time.time()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion1 = nn.CrossEntropyLoss(reduction='none').cuda()
        criterion2 = nn.CrossEntropyLoss().cuda()
        beta=1.0
        participation_rate=0.4
        bs = 128 #self.sleep_bs
        num_epochs = self.sleep_epoch
        num_iter = math.ceil(len(rehearsal_ixs) / bs)
        print('Number of Training Samples:', len(rehearsal_ixs))
        loss_arr = np.zeros(num_iter, np.float32)
        for epoch in range(num_epochs):
            # Gather previous data for replay
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
                end = (i + 1) * bs
                if start > (y.shape[0] - bs):
                    start = y.shape[0] - bs
                    end = y.shape[0]
                codes_batch = X[start:end]
                labels_batch = y[start:end].cuda()
                # Reconstruct/decode samples with PQ
                codes_batch = np.reshape(codes_batch, (
                    codes_batch.shape[0] * self.num_feats * self.num_feats, self.num_codebooks)) # 256N x 8                    
                data_batch_recon = pq.decode(codes_batch) #pq # 256N x 128
                data_batch_recon = opq.reverse_transform(data_batch_recon) #opq # 256N x 128
                data_batch_recon = np.reshape(data_batch_recon,
                        (-1, self.num_feats, self.num_feats, self.num_channels))  # 128 x 16 x 16 x 128
                data_batch_recon = torch.from_numpy(np.transpose(data_batch_recon, (0, 3, 1, 2))).cuda()  # 128 x 128 x 16 x 16

                if np.random.rand(1) < participation_rate:
                    ### MIXUP: Do mixup between two batches of previous data
                    num_instances = math.ceil(data_batch_recon.shape[0]/2) # 64
                    x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                        data_batch_recon[:num_instances], labels_batch[:num_instances],
                        data_batch_recon[num_instances:], labels_batch[num_instances:],
                        alpha=0.1)
                    data = torch.empty((num_instances, self.num_channels, self.num_feats, self.num_feats))  # 64 x 128 x 16 x 16
                    data = x_prev_mixed.clone()  # 64 x 128 x 16 x 16
                    labels_a = torch.zeros(num_instances).long()  # 64
                    labels_b = torch.zeros(num_instances).long()  # 64
                    labels_a = prev_labels_a
                    labels_b = prev_labels_b
                    output = classifier_F(data.cuda())  # 64 x 128 x 16 x 16
                    ### Manifold MixUp
                    loss = self.mixup_criterion(criterion1, output, labels_a.cuda(), labels_b.cuda(), lam)
                    loss = loss.mean()
                else:
                    ##### /// CutMix /// #######
                    num_instances = math.ceil(data_batch_recon.shape[0]/2) # 64
                    input_a = data_batch_recon[:num_instances] # first 64
                    input_b = data_batch_recon[num_instances:] # last 64
                    lam = np.random.beta(beta, beta)
                    target_a = labels_batch[:num_instances] # first 64
                    target_b = labels_batch[num_instances:] # last 64
                    bbx1, bby1, bbx2, bby2 = self.rand_bbox(input_a.size(), lam)
                    input_a[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_a.size()[-1] * input_a.size()[-2]))
                    # compute output
                    output = classifier_F(input_a)
                    loss = criterion2(output, target_a) * lam + criterion2(output, target_b) * (1. - lam)

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
                'best_acc5': best_acc5,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                }, is_best, save_dir, ckpt_file)
        print('\nTime Spent in base initialization (in mins): %0.3f' % ((time.time() - start_time)/60))
    
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

    def rand_bbox_thumb(inputs_size, dst_size):
        x = random.randint(0, inputs_size-dst_size)
        y = random.randint(0, inputs_size-dst_size)
        return x, y, x+dst_size, y+dst_size
    
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

    def save(self, inc, save_full_path, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, opq, pq):
        if not os.path.exists(save_full_path):
            os.makedirs(save_full_path)
        state = {
            'model_state_dict': self.classifier_F.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print(f'\nSaving DNN model to {save_full_path}')
        torch.save(state, os.path.join(save_full_path, 'classifier_F_%d.pth' % inc))
        ## get OPQ parameters
        d=self.num_channels
        A = faiss.vector_to_array(opq.A).reshape(d, d) ## from Swig Object to np array
        b = faiss.vector_to_array(opq.b) ## from Swig Object to np array
        ## get PQ centroids/codebooks
        centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
        ## save in a dictionary
        d = {'latent_dict': latent_dict, 'rehearsal_ixs': rehearsal_ixs,
             'class_id_to_item_ix_dict': class_id_to_item_ix_dict, 
             'opq_A': A, 'opq_b': b, 'pq_centroids': centroids}
        with open(os.path.join(save_full_path, 'buffer_%d.pkl' % inc), 'wb') as f:
            pickle.dump(d, f)

    def resume(self, inc, resume_full_path):
        print(f'\nResuming DNN model from {resume_full_path}')
        state = torch.load(os.path.join('./' + resume_full_path, 'best_' + resume_full_path + '.pth'))
        utils.safe_load_dict(self.classifier_F, state['state_dict'], should_resume_all_params=False)
        ## sanity check whether two checkpoints are similar ##
        old_state=state['state_dict']
        new_state = self.classifier_F.state_dict()
        for k in old_state:
            #assert torch.equal(old_state[k].cpu(), new_state[k[len("module."):]]), k
            assert torch.equal(old_state[k].cpu(), new_state[k].cpu()), k
        print("Successfully performed sanity check!!")
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
    
    def resume0(self, inc, resume_full_path):
        print(f'\nResuming DNN model from {resume_full_path}')
        #state = torch.load(os.path.join('./' + resume_full_path, 'best_' + resume_full_path + '.pth'))
        state = torch.load(os.path.join(resume_full_path, 'classifier_F_%d.pth' % inc))
        utils.safe_load_dict(self.classifier_F, state['model_state_dict'], should_resume_all_params=False)
        #self.optimizer.load_state_dict(state['optimizer'])
        #utils.safe_load_dict(self.classifier_F, state['state_dict'], should_resume_all_params=False)
        ## sanity check whether two checkpoints are similar ##
        #old_state=state['state_dict']
        old_state=state['model_state_dict']
        new_state = self.classifier_F.state_dict()

        for k in old_state:
            #assert torch.equal(old_state[k].cpu(), new_state[k[len("module."):]]), k
            assert torch.equal(old_state[k].cpu(), new_state[k].cpu()), k
        print("Successfully performed sanity check!!")

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

        ### THE END ###