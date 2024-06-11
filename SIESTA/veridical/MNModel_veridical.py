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
import math
import pickle
import copy
import utils_gd as utils
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
    def __init__(self, num_classes, classifier_G='MobNetClassifyAfterLayer8',
            extract_features_from='model.features.7', classifier_F='MobNet_StartAt_Layer8', classifier_ckpt=None, 
            weight_decay=1e-5, lr_mode=None, lr_step_size=100, start_lr=0.1, end_lr=0.001, lr_gamma=0.5, num_samples=50, 
            sleep_samples=50000, mixup_alpha=0.1, grad_clip=None, num_channels=80, num_feats=14, 
            num_codebooks=8, codebook_size=256, max_buffer_size=None, 
            sleep_batch_size=128, sleep_epoch=50, sleep_lr=0.2):
        # make the classifier
        self.classifier_F = utils.build_classifier(classifier_F, classifier_ckpt, num_classes=num_classes)
        #core_model = utils.build_classifier(classifier_G, classifier_ckpt, num_classes=None)
        #self.classifier_G = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

        # OPTIMIZER
        params = self.get_layerwise_params(self.classifier_F, sleep_lr)
        self.optimizer = optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
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
        self.sleep_bs = sleep_batch_size
        self.sleep_epoch = sleep_epoch
        ## SLDA
        self.slda = StreamingLDA(1280, num_classes, shrinkage_param=1e-2) # >> optimal
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
    

    ##### ------------------------------- #####
    ##### ----- UPDATE INDEX-BUFFER ----- #####
    ##### ------------------------------- #####
    def update_buffer(self, curr_loader, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter):
        start_time = time.time()
        classifier_F = self.classifier_F.cuda()
        classifier_F.eval()
        start_ix=0
        recent_lbls = np.zeros((len(curr_loader.dataset)))
        with torch.no_grad():
            ### ------ New Samples ------ ###
            for batch_images, batch_labels, batch_item_ixs in curr_loader: # New classes
                end_ix = start_ix + batch_labels.shape[0]
                recent_lbls[start_ix:end_ix] = batch_labels.squeeze()
                start_ix = end_ix
                # get penultimate features from network
                data_batch = classifier_F.get_penultimate_feature(batch_images.cuda()) # N x 1280 ## pre-activation
                # put codes and labels into buffer (dictionary)
                for x, y, item_ix in zip(data_batch, batch_labels, batch_item_ixs):
                    ## Fit SLDA to new data
                    self.slda.fit(x, y.view(1, ), sleep=False) # False for not updating Cov matrix
                    ## Add index of new data point to rehearsal_ixs (new class)
                    latent_dict[int(item_ix.numpy())] = [y.numpy()]
                    rehearsal_ixs.append(int(item_ix.numpy()))
                    class_id_to_item_ix_dict[int(y.numpy())].append(int(item_ix.numpy()))
                    # if buffer is full, randomly replace previous example from class with most samples
                    if self.max_buffer_size is not None and counter.count >= self.max_buffer_size:
                        ## class with most samples and random item_ix from it
                        max_key = max(class_id_to_item_ix_dict, key=lambda x: len(class_id_to_item_ix_dict[x]))
                        max_class_list = class_id_to_item_ix_dict[max_key]
                        rand_item_ix = random.choice(max_class_list) # randomly remove sample from largest class
                        ## remove the random_item_ix from all buffer references
                        max_class_list.remove(rand_item_ix)
                        latent_dict.pop(rand_item_ix)
                        rehearsal_ixs.remove(rand_item_ix)
                    else:
                        counter.update()
            spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("Time spent in buffer update process (in mins):", spent_time)
        recent_class_list = np.unique(recent_lbls)
        return latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list

    
    ### ---------------------------- ###
    ### ----- Wake Update Rule ----- ###
    ### ---------------------------- ###
    def update_new_nodes(self, class_list):
        start_time = time.time()
        bias=torch.ones(1).cuda()
        for k in class_list:
            k = torch.tensor(k, dtype=torch.int32)
            mu_k = self.slda.grab_mean(k)
            self.classifier_F.state_dict()['model.classifier.3.weight'][k].copy_(mu_k)
            self.classifier_F.state_dict()['model.classifier.3.bias'][k] = bias
        print('Elapsed time for updating new nodes (in SEC): %0.3f' % (time.time() - start_time))


    ### ---------------------------------------------------- ###
    ### ---------- extract penultimate features ------------ ###
    ### ---------------------------------------------------- ###
    def get_feature(self, data_loader):
        with torch.no_grad():
            classifier_F = self.classifier_F.cuda()
            classifier_F.eval()
            all_feat = torch.zeros((len(data_loader.dataset), 1280), dtype=torch.float64)
            all_lbls = torch.zeros((len(data_loader.dataset)))
            all_ixs = torch.zeros((len(data_loader.dataset)))
            start_ix = 0
            for batch_images, batch_labels, batch_item_ixs in data_loader:
                end_ix = start_ix + batch_labels.shape[0]
                ### get penultimate features from network
                feat_batch = classifier_F.get_penultimate_feature(batch_images.cuda()) # N x 1280 ## pre-activation
                all_feat[start_ix:end_ix] = feat_batch
                all_lbls[start_ix:end_ix] = batch_labels.squeeze()
                all_ixs[start_ix:end_ix] = batch_item_ixs
                start_ix = end_ix
        return all_feat, all_lbls, all_ixs


    def dist_to_centroid(self, data_loader):
        start_time = time.time()
        dist_dict = {}
        model_clone = copy.deepcopy(self.classifier_F)
        model_clone.cuda()
        model_clone.eval()
        all_feat, all_lbls, all_ixs = self.get_feature(data_loader)
        class_list = np.unique(all_lbls)
        for c in class_list:
            class_feat = all_feat[all_lbls == c]  # filter features by class c # NC x 1280
            class_ixs = all_ixs[all_lbls == c]
            mean_feature = torch.mean(class_feat, 0) # mean across samples i.e., row wise NCx1280 -> 1280
            class_feat = class_feat.detach().cpu().numpy()
            mean_feature = mean_feature.detach().cpu().numpy()
            mean_feature = np.reshape(mean_feature, (-1, 1280))
            dist = cosine_distances(mean_feature, class_feat) # 1 x NC
            dist = dist.squeeze()
            class_ixs = np.array(class_ixs, dtype=np.int32) # important for speed up!
            dist_dict[c] = [dist, class_ixs]
            ts = int((time.time() - start_time)/60) # in mins
        print("Elasped time in distance computation is:", ts)
        return dist_dict, class_list


    ## GRASP Sampling ##
    def grasp_sampling(self, data_loader, num_iters):
        dist_dict, class_list = self.dist_to_centroid(data_loader)
        rehearsal_idxs = []
        count = 0
        budget = self.sleep_bs*num_iters # 512*iters

        while count < budget:
            for i in range(len(class_list)):
                c = class_list[i] # class
                dist_current_class = dist_dict[c][0] # grabing distances of all samples of class c
                ixs_current_class = dist_dict[c][1] # grabing indices of all samples of class c
                probas = np.array(dist_current_class) # > hard examples
                probas = 1 / (probas + 1e-7) # min distances get highest scores/ priorities > easy examples
                p_curr_class = probas / np.linalg.norm(probas, ord=1)  # sum to 1
                ## sample
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


    ### --------------------------- ###
    ### /// Min Replay Sampling /// ###
    ### --------------------------- ###
    def sample_min_replay(self, latent_dict, rehearsal_ixs, new_class_list,
        replay_count=None, replay_ixs=None, step=0):
        start_time = time.time()
        keep = self.sleep_bs
        if step == 0:
            new_count = 1
            old_count = 50
            replay_count = []
            replay_ixs = []
            for ii, v in enumerate(rehearsal_ixs):
                label = torch.from_numpy(latent_dict[v][0])
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

        
    def rehearsal_minrep(self, latent_dict, rehearsal_ixs, new_class_list, num_iters):
        start_time = time.time()
        replay_idxs = []
        budget = self.sleep_bs*num_iters #256*num_iters
        _, replay_count, rep_ixs, ts = self.sample_min_replay(latent_dict, rehearsal_ixs, new_class_list)
        sampling_time = ts
        for i in range(num_iters):
            chosen_rehearsal_ixs, replay_count, rep_ixs, ts = self.sample_min_replay(latent_dict, 
            rehearsal_ixs, new_class_list, replay_count, rep_ixs, i+1)
            replay_idxs.append(torch.from_numpy(chosen_rehearsal_ixs))
            sampling_time += ts
        ##
        replay_idxs = torch.cat(replay_idxs, dim=0)
        #replay_idxs = torch.tensor(replay_idxs[:budget], dtype=torch.int32)
        replay_idxs = torch.tensor(replay_idxs, dtype=torch.int32)
        assert len(replay_idxs) == budget
        replay_idxs = np.array(replay_idxs, dtype=np.int32)
        spent_time = int((time.time() - start_time) / 60) - int(sampling_time/60)  # in minutes
        print("Number of selected samples by min rehearsal:", len(replay_idxs))
        print("\nTime Spent in Updating DNN (in Mins):", spent_time)
        return replay_idxs

    
    def uniform_balanced(self, latent_dict, rehearsal_ixs, num_iters):
        ixs = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        lbl = torch.empty(len(rehearsal_ixs), dtype=torch.long) # N
        for ii, v in enumerate(rehearsal_ixs):
            lbl[ii] = torch.from_numpy(latent_dict[v][0])
            ixs[ii] = v
        class_list = np.unique(lbl)
        replay_idxs = []
        k=1
        count = 0
        budget = self.sleep_bs*num_iters #64*num_iters
        while count < budget:
            for c in class_list:
                ixs_current_class = ixs[lbl == c]
                sel_idx = np.random.choice(ixs_current_class, size=k)
                count += k
                replay_idxs.append(torch.from_numpy(sel_idx))
                if count >= budget:
                    break
        replay_idxs = torch.cat(replay_idxs, dim=0)
        replay_idxs = torch.tensor(replay_idxs[:budget], dtype=torch.int32)
        #print("Num of samples:", len(replay_idxs))
        assert len(replay_idxs) <= budget
        replay_idxs = np.array(replay_idxs, dtype=np.int32)
        return replay_idxs

    
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
    

    def decode(self, data_batch, classifier_F):
        new_data = classifier_F.get_penultimate_feature(data_batch) # N x 1280 ## pre-activation
        return new_data


    ### ---------------------------------- ###
    ### -------- SIESTA Rehearsal --------- ###
    ### ---------------------------------- ###
    def siesta_rehearsal(self, replay_loader, num_iters):
        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        print("Not using any feature augmentations..")
        num_iter = num_iters
        optimizer2 = optim.SGD(params, momentum=0.9, weight_decay=1e-5)
        lr_scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, 
            max_lr=self.sleep_lr, steps_per_epoch=num_iters, epochs=1) ## OneCycle

        for batch_ix, batch in enumerate(replay_loader):
            batch_x, batch_lbls = batch[0], batch[1]
            batch_x = batch_x.cuda()
            batch_lbls = batch_lbls.cuda()
            ### fit on replay mini-batch
            output = classifier_F(batch_x)
            loss = criterion(output, batch_lbls)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            ### Update lr scheduler
            lr_scheduler2.step()
            total_loss.update(loss.item())

            if (batch_ix+1) % 5000 == 0 or batch_ix == 0 or (batch_ix+1) == num_iter:
                print("Iter:", (batch_ix+1), "-- Loss: %1.5f" % total_loss.avg)
        spent_time = int((time.time() - start_time) / 60) # in mins
        print("\nTime Spent in Updating DNN (in mins):", spent_time)


    # ------------------------------------------------------------------- #
    # ------ Joint Train on Base Data using feature augmentations ------- #
    # ------------------------------------------------------------------- #
    def joint_train_base(self, train_loader, test_loader, save_dir, ckpt_file):
        print("Offline training mobilenet using image augmentations..")
        best_acc1=0
        best_acc5=0
        start_time = time.time()
        #classifier_F = nn.DataParallel(self.classifier_F).cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        num_epochs = self.sleep_epoch
        num_iter = math.ceil(len(train_loader.dataset) / self.sleep_bs)
        print('Number of Training Samples:', len(train_loader.dataset))

        for epoch in range(num_epochs):
            loss_arr = np.zeros(num_iter, np.float32)

            for batch_ix, batch in enumerate(train_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()
                batch_lbls = batch_lbls.cuda()
                output = classifier_F(batch_x)
                loss = criterion(output, batch_lbls)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_arr[batch_ix] = loss.item()

            ### Update lr scheduler
            self.lr_scheduler.step()
            
            if (epoch + 1) % 1 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                probas, true = self.predict(test_loader)
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

    
    ### --------------------------------- ###
    ### ---------- Inference ------------ ###
    ### --------------------------------- ###
    def predict(self, data_loader):
        with torch.no_grad():
            self.classifier_F.eval()
            self.classifier_F.cuda()
            probas = torch.zeros((len(data_loader.dataset), self.num_classes), dtype=torch.float64)
            all_lbls = torch.zeros((len(data_loader.dataset)))
            start_ix = 0
            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()
                batch_lbls = batch_lbls.cuda()
                logits = self.classifier_F(batch_x)
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
        old_state=state['state_dict']
        new_state = self.classifier_F.state_dict()
        for k in old_state:
            assert torch.equal(old_state[k].cpu(), new_state[k[len("module."):]]), k
        print("Successfully performed sanity check!!")
        # load parameters
        #with open(os.path.join(resume_full_path, 'buffer_%d.pkl' % inc), 'rb') as f:
        #    d = pickle.load(f)
        #nbits = int(np.log2(self.codebook_size))
        #pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)
        #opq = faiss.OPQMatrix(self.num_channels, self.num_codebooks)
        #opq.pq = pq
        #faiss.copy_array_to_vector(d['pq_centroids'].ravel(), pq.centroids)
        #faiss.copy_array_to_vector(d['opq_A'].ravel(), opq.A)
        #faiss.copy_array_to_vector(d['opq_b'].ravel(), opq.b)
        #opq.is_trained = True
        #opq.is_orthonormal = True
        #return d['latent_dict'], d['rehearsal_ixs'], d['class_id_to_item_ix_dict'], opq, pq

        ### THE END ###