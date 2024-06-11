import argparse
import torch
import json
import os
import numpy as np
from MNModel_LT import MNModel
#import wandb
from places_base_init_pq import *
torch.multiprocessing.set_sharing_strategy('file_system')

### MobileNet / SIESTA / Places-LT-365 / Latent Rehearsal

def get_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, seed=1, return_item_ix=False):
    data_loader = utils_places.get_places_data_loader(images_dir + '/' + split, label_dir, split, batch_size=batch_size, 
            shuffle=False, min_class=min_class, max_class=max_class, seed=seed, return_item_ix=return_item_ix)
    return data_loader

def streaming(args, mnet):
    accuracies = {'base_classes_top1': [], 'non_base_classes_top1': [], 'seen_classes_top1': [],
                  'base_classes_top5': [], 'non_base_classes_top5': [], 'seen_classes_top5': []}
    counter = utils.Counter()
    start_time=time.time()
    num_sleep =0
    rotation =0 # to alternate/ control wake/sleep cycle
    sleep_freq = 2 # for wake/sleep cycle one after another repeatedly

    if args.resume_full_path is not None:
        # load in previous model to continue training
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, opq, pq = mnet.resume(args.base_init_classes, 
                   args.resume_full_path)
        latent_dict = {}
        class_id_to_item_ix_dict = defaultdict(list)
        rehearsal_ixs = []
    else:
        print('\nPerforming base initialization...')
        feat_data, label_data, item_ix_data = extract_base_init_features(args.images_dir, args.label_dir,
                                    args.extract_features_from, args.classifier_ckpt, args.base_arch, 
                                    args.base_init_classes, args.num_channels, args.spatial_feat_dim)

        pq, opq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = fit_opq(feat_data, label_data, item_ix_data,
                args.num_channels, args.spatial_feat_dim, args.num_codebooks, args.codebook_size, counter=counter)
        print('Counter after pq init.', counter.count)
        ################
        ## INITIALIZING REMIND (fc IN CLASSIFIER_F) WITH BASE INIT CLASSES
        start_time_init = time.time()
        print('\nTraining classes {}-{}.'.format(args.min_class, args.base_init_classes))
        ##### Joint base-init training (ONlY APPLICABLE FOR SSL PRETRAINED MODEL) #####
        init_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', min_class=args.min_class,
                                max_class=args.base_init_classes, batch_size=args.batch_size, seed=args.seed)
        mnet.joint_train_base(opq, pq, latent_dict, rehearsal_ixs, init_test_loader, args.save_dir, args.ckpt_file)
        ### save mnet model
        #mnet.save(args.base_init_classes, args.save_dir, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, opq, pq)
        ################
        
        ###INFERENCE ON BASE-INIT CLASSES
        print('\nComputing base accuracies...')
        probas_base, true_base = mnet.predict(init_test_loader, opq, pq)
        top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
        print('\nDNN Accuracy on BASE-INIT Classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))
        print('\nRuntime for current iteration (in mins): %0.3f' % ((time.time() - start_time_init)/60))

    all_class_list = []
    old_class_list = []
    last_class = 365
    sleep_tot = int((last_class - args.streaming_min_class) / args.class_increment) ## 5 for inc=73
    print("Total number of sleeps for Places-LT:", sleep_tot)
    val_acc_all=[]
    val_acc_old=[]
    val_acc_new=[]
    print('\nBeginning SIESTA Training...')
    new_class_list = []
    num_recent_stuff = 0
    num_iter= int(600 * (64 / args.sleep_batch_size))
    print("Total number of iterations per rehearsal cycle is :", num_iter)
    for class_ix in range(args.streaming_min_class, args.streaming_max_class, args.class_increment):
        max_class = class_ix + args.class_increment # 0-73, 73-146, 146-219, 219-292, 292-365
        start_time_itr = time.time()
        rotation += 1
        print('\nTraining classes {}-{}.'.format(class_ix, max_class))

        train_loader_curr = get_data_loader(args.images_dir, args.label_dir, 'train', class_ix, max_class,
                batch_size=args.batch_size, seed=args.seed, return_item_ix=True) # 0-73, 73-146, 146-219, 219-292, 292-365
        
        ### store current data in memory buffer        
        print('Number of stored samples BEFORE buffer update:', len(rehearsal_ixs))      
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list =mnet.update_buffer(opq, pq, 
            train_loader_curr, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter)
        print('Counter AFTER buffer update.', counter.count)
        new_class_list = np.append(new_class_list, recent_class_list)
        num_recent_stuff = len(train_loader_curr.dataset) # 73
        all_class_list = np.append(all_class_list, recent_class_list)
        
        #### ---------------------- AWAKE PHASE ------------------------####
        print('\nSTATUS -> AWAKE phase ...')

        mnet.update_new_nodes(recent_class_list)

        ## All, Old & New Classes
        test_loader_all = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size, seed=args.seed) # 0-max
        if max_class == args.class_increment:
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size, seed=args.seed) # 0-73
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size, seed=args.seed) # 0-73
        else:
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, class_ix,
                                    batch_size=args.batch_size, seed=args.seed) # 0-73
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', class_ix, max_class,
                                    batch_size=args.batch_size, seed=args.seed) # 73-146
        ### DNN ###
        print("\nWake Evaluation")
        probas_all, true_all = mnet.predict(test_loader_all, opq, pq)
        top1_all, top5_all = utils.accuracy(probas_all, true_all, topk=(1, 5))
        print('\nWake Online Performance [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_all, top5_all))
        probas_old, true_old = mnet.predict(test_loader_old, opq, pq)
        top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
        print('\nWake Online Performance [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
        probas_new, true_new = mnet.predict(test_loader_new, opq, pq)
        top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
        print('\nWake Online Performance [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))

        val_acc_all = np.append(val_acc_all, top1_all)
        val_acc_old = np.append(val_acc_old, top1_old)
        val_acc_new = np.append(val_acc_new, top1_new)
        ################
        #///////////////////////////////////////////////////// SLEEP PHASE /////////////////////////////////////////////
        ### Train DNN
        #print("Rehearsal Strategy is Uniform Balanced")
        #mnet.rehearsal_uniform_bal(opq, pq, latent_dict, rehearsal_ixs, num_iter)

        print("Rehearsal Strategy is GRASP")
        mnet.rehearsal_grasp(opq, pq, latent_dict, rehearsal_ixs, num_iter)

        print("\nEvaluation..")
        # All Seen Classes
        probas_all, true_all = mnet.predict(test_loader_all, opq, pq)
        top1_all, top5_all = utils.accuracy(probas_all, true_all, topk=(1, 5))
        print('Post-sleep accuracy [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_all, top5_all))
        ## OLD Classes
        probas_old, true_old = mnet.predict(test_loader_old, opq, pq)
        top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
        print('Post-sleep accuracy [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
        ## NEW Classes
        probas_new, true_new = mnet.predict(test_loader_new, opq, pq)
        top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
        print('Post-sleep accuracy [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))

        val_acc_all = np.append(val_acc_all, top1_all)
        val_acc_old = np.append(val_acc_old, top1_old)
        val_acc_new = np.append(val_acc_new, top1_new)

        ##### Reset New Class List #####
        old_class_list = np.append(old_class_list, new_class_list) ## After sleep new classes become old class!
        new_class_list = []
        num_recent_stuff = 0
        ####### /// #######

        #if max_class == last_class:
        #    break

    ###save final trained model
    mnet.save(args.seed, args.save_dir)
    
    exp_dir = args.save_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    filename1 = 'grasp_val_all_top1_seed_' +str(args.seed) + '.npy'
    filename2 = 'grasp_val_old_top1_seed_' +str(args.seed) + '.npy'
    filename3 = 'grasp_val_new_top1_seed_' +str(args.seed) + '.npy'
    np.save(os.path.join(exp_dir, filename1), val_acc_all)
    np.save(os.path.join(exp_dir, filename2), val_acc_old)
    np.save(os.path.join(exp_dir, filename3), val_acc_new)
    print('\nRuntime for entire experiment (in mins): %0.3f' % ((time.time() - start_time)/60))
    ## /// END /// ##

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # directories and names
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--label_dir', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir', type=str, default=None)  # directory for places train/val folders
    parser.add_argument('--save_dir', type=str, required=False)  # directory for saving results
    parser.add_argument('--resume_full_path', type=str, default=None)  # directory of previous model to load
    parser.add_argument('--ckpt_file', type=str, default='MobileNet_modified_100c.pth')
    # network parameters
    parser.add_argument('--base_arch', type=str, default='MobNetClassifyAfterLayer8')  # architecture for G
    parser.add_argument('--classifier', type=str, default='MobNet_StartAt_Layer8')  # architecture for F
    parser.add_argument('--classifier_ckpt', type=str, required=True)  # base-init ckpt ///// Pretrain Checkpoint /////
    parser.add_argument('--extract_features_from', type=str,
                        default='model.features.7')  # name of the layer to extract features              
    parser.add_argument('--num_channels', type=int, default=80)  # number of channels where features are extracted
    parser.add_argument('--spatial_feat_dim', type=int, default=14)  # spatial dimension of features being extracted
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # weight decay for network
    parser.add_argument('--batch_size', type=int, default=128)  # testing batch size
    # pq parameters
    parser.add_argument('--num_codebooks', type=int, default=8)
    parser.add_argument('--codebook_size', type=int, default=256)
    # replay buffer parameters
    parser.add_argument('--rehearsal_samples', type=int, default=50)  # number of replay samples
    parser.add_argument('--max_buffer_size', type=int, default=None)  # maximum number of samples in buffer
    # learning rate parameters
    parser.add_argument('--lr_mode', type=str, choices=['step_lr_per_class'],
                        default='step_lr_per_class')  # decay the lr per class
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--start_lr', type=float, default=0.1)  # starting lr for class
    parser.add_argument('--end_lr', type=float, default=0.001)  # ending lr for class

    # augmentation parameters
    #parser.add_argument('--use_random_resized_crops', action='store_true')
    #parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.1)
    # streaming setup
    parser.add_argument('--num_classes', type=int, default=365)  # total number of classes
    parser.add_argument('--min_class', type=int, default=0)  # overall minimum class
    parser.add_argument('--base_init_classes', type=int, default=100)  # number of base init classes
    parser.add_argument('--class_increment', type=int, default=100)  # how often to evaluate
    parser.add_argument('--streaming_min_class', type=int, default=100)  # class to begin stream training
    parser.add_argument('--streaming_max_class', type=int, default=1000)  # class to end stream training
    parser.add_argument('--sleep_batch_size', type=int, default=64)  # training batch size during sleep
    parser.add_argument('--sleep_epoch', type=int, default=50)  # training epoch during sleep
    parser.add_argument('--sleep_lr', type=float, default=0.2)  # starting lr for sleep offline training
    parser.add_argument('--seed', type=int, default=1)

    # get arguments and print them out and make any necessary directories
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.lr_mode == 'step_lr_per_class':
        args.lr_gamma = np.exp(args.lr_step_size * np.log(args.end_lr / args.start_lr) / 1300)

    #print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # make model and begin stream training
    mnet = MNModel(num_classes=args.num_classes, classifier_G=args.base_arch,
                    extract_features_from=args.extract_features_from, classifier_F=args.classifier,
                    classifier_ckpt=args.classifier_ckpt,
                    weight_decay=args.weight_decay, lr_mode=args.lr_mode, lr_step_size=args.lr_step_size,
                    start_lr=args.start_lr, end_lr=args.end_lr, lr_gamma=args.lr_gamma,
                    num_samples=args.rehearsal_samples,
                    mixup_alpha=args.mixup_alpha, grad_clip=None, 
                    num_channels=args.num_channels, num_feats=args.spatial_feat_dim,
                    num_codebooks=args.num_codebooks, codebook_size=args.codebook_size,
                    max_buffer_size=args.max_buffer_size, sleep_batch_size=args.sleep_batch_size, 
                    sleep_epoch=args.sleep_epoch, sleep_lr=args.sleep_lr, seed=args.seed)
    streaming(args, mnet)