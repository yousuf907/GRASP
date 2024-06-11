import argparse
import torch
import json
import os
import numpy as np
from MNModel_veridical import MNModel
from imagenet_base_init import *
torch.multiprocessing.set_sharing_strategy('file_system')

### Architecture: MobileNet / Dataset: ImageNet / Algorithm: SIESTA with Veridical Rehearsal ###

def get_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, return_item_ix=False):
    data_loader = utils_imagenet2.get_imagenet_data_loader(images_dir + '/' + split, label_dir, split,
        batch_size=batch_size, shuffle=False, min_class=min_class, max_class=max_class, return_item_ix=return_item_ix)
    return data_loader

def get_replay_loader(images_dir, label_dir, split, sel_idxs, batch_size=128, return_item_ix=False):
    replay_loader = utils_imagenet2.get_imagenet_replay_loader(images_dir + '/' + split, label_dir, split, sel_idxs,
        batch_size=batch_size, shuffle=False, return_item_ix=return_item_ix)
    return replay_loader

def streaming(args, mnet):
    counter = utils.Counter()
    start_time=time.time()
    num_sleep = 0
    rotation = 0 # to alternate/ control wake/sleep cycle
    sleep_freq = 2 # for wake/sleep cycle one after another repeatedly
    latent_dict = {}
    seen_class_list = []
    rehearsal_ixs = []
    class_id_to_item_ix_dict = defaultdict(list)
    class_id_dist = defaultdict(list)

    base_init_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class,
            args.base_init_classes, batch_size=args.batch_size) #0-100

    print('\nComputing base accuracies...')
    probas_base, true_base = mnet.predict(base_init_test_loader)
    top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
    print('\nAccuracy on BASE-INIT Classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))

    ### fill buffer with base init data
    train_loader_curr = get_data_loader(args.images_dir, args.label_dir, 'train', args.min_class,
            args.base_init_classes, batch_size=args.batch_size, return_item_ix=True) # 0-100
    latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list = mnet.update_buffer(train_loader_curr,
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter)  # 0-100
    print('Counter AFTER buffer update.', counter.count)
    seen_class_list = np.append(seen_class_list, recent_class_list)
    #exit()

    ##
    last_class=1000
    sleep_tot = int((last_class - args.base_init_classes) / (2 * args.class_increment)) ## 9 for inc=50
    print("Total number of rehearsal cycles for ImageNet-1K:", sleep_tot)
    print('\nBeginning GDumb Training...')
    new_class_list = []
    num_recent_stuff = 0
    ###
    val_acc_all=[]
    val_acc_old=[]
    val_acc_new=[]
    val_acc_all = np.append(val_acc_all, top1_base)
    val_acc_old = np.append(val_acc_old, top1_base)
    val_acc_new = np.append(val_acc_new, top1_base)
    dnn_acc_all=[]
    dnn_acc_old=[]
    dnn_acc_new=[]
    dnn_acc_all = np.append(dnn_acc_all, top1_base)
    dnn_acc_old = np.append(dnn_acc_old, top1_base)
    dnn_acc_new = np.append(dnn_acc_new, top1_base)
    ### Iterations per Sleep
    num_iter= int(20017 * (64 / args.sleep_batch_size) * (args.class_increment / 50))
    print("Number of fixed iterations per sleep:", num_iter)
    Wt=0
    St0=0
    St=0

    for class_ix in range(args.streaming_min_class, args.streaming_max_class, args.class_increment):
        max_class = class_ix + args.class_increment  # 0-50, 50-100, 150-200
        start_time_itr = time.time()
        rotation += 1
        print('\nTraining classes {}-{}.'.format(class_ix, max_class))
        train_loader_curr = get_data_loader(args.images_dir, args.label_dir, 'train', class_ix, max_class,
                    batch_size=args.batch_size, return_item_ix=True) # 0-50, 50-100, 150-200

        ### store current data in memory buffer
        print('Number of stored samples BEFORE buffer update:', len(rehearsal_ixs))
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list = mnet.update_buffer(train_loader_curr,
            latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter)  # 0-50, 50-100, 150-200
        print('Counter AFTER buffer update.', counter.count)
        new_class_list = np.append(new_class_list, recent_class_list)
        num_recent_stuff = len(train_loader_curr.dataset)
        seen_class_list = np.append(seen_class_list, recent_class_list)

        #### AWAKE PHASE ####
        if rotation < sleep_freq: # 1 < 2
            print('\nSTATUS -> Online/ Awake Phase...')
            start_time_wake = time.time()
            ## wake updates
            mnet.update_new_nodes(recent_class_list)

            ## All & New Classes
            test_loader_all = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size) # 0-125
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', class_ix, max_class,
                                    batch_size=args.batch_size) # 100-125
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, class_ix,
                                    batch_size=args.batch_size) # 0-100
            ### DNN ###
            probas_dnn, true_dnn = mnet.predict(test_loader_all)
            top1_wake, top5_wake = utils.accuracy(probas_dnn, true_dnn, topk=(1, 5))
            print('\nWake Online Performance [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_wake, top5_wake))
            ## OLD CLASSES
            probas_old, true_old = mnet.predict(test_loader_old)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print('Wake Online Performance [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
            ## NEW CLASSES
            probas_new, true_new = mnet.predict(test_loader_new)
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print('Wake Online Performance [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))

            val_acc_all = np.append(val_acc_all, top1_wake)
            val_acc_old = np.append(val_acc_old, top1_old)
            val_acc_new = np.append(val_acc_new, top1_new)
            print('\nTime Spent in Recent Wake (in MINs): %0.3f' % ((time.time() - start_time_wake)/60))


        #/////////////////////////////////////////////// SLEEP PHASE /////////////////////////////////////////////
        elif rotation >= sleep_freq: # 2 >= 2
            start_time_sleep = time.time()
            rotation = 0
            num_sleep += 1
            print('\nSTATUS -> SLEEP... Number of Sleeps =', num_sleep)
            old_class_ix = max_class - len(new_class_list)

            ## All classes after sleep
            test_loader_all = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size) # 0-200
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, old_class_ix,
                                    batch_size=args.batch_size) # 0-100
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', old_class_ix, max_class,
                                    batch_size=args.batch_size) # 100-200

            assert len(new_class_list) == 2 * args.class_increment

            start_time_w = time.time()
            ### Wake updates ###
            mnet.update_new_nodes(recent_class_list)


            print("Pre-Sleep Evaluations..")
            ### All Classes ###
            probas_dnn, true_dnn = mnet.predict(test_loader_all)
            top1_wake, top5_wake = utils.accuracy(probas_dnn, true_dnn, topk=(1, 5))
            print('\nWake Online Performance [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_wake, top5_wake))
            ### OLD CLASSES ###
            probas_old, true_old = mnet.predict(test_loader_old)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print('Wake Online Performance [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
            old_class_list = np.unique(true_old)
            ### NEW CLASSES ###
            probas_new, true_new = mnet.predict(test_loader_new)
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print('Wake Online Performance [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))

            val_acc_all = np.append(val_acc_all, top1_wake)
            val_acc_old = np.append(val_acc_old, top1_old)
            val_acc_new = np.append(val_acc_new, top1_new)
            print('\nTime Spent in Recent Wake (in MINs): %0.3f' % ((time.time() - start_time_w)/60))


            ### Rehearsal during Sleep ###
            ### Min Rehearsal ###
            #print("Replay Strategy is Min Rehearsal.")
            ##sel_idxs = mnet.rehearsal_minrep(rehearsal_ixs, num_iter)
            #sel_idxs = mnet.rehearsal_minrep(latent_dict, rehearsal_ixs, new_class_list, num_iter)

            ## uniform class-balanced ###
            #print("Replay Strategy is Uniform Class-Balanced.")
            #sel_idxs = mnet.uniform_balanced(latent_dict, rehearsal_ixs, num_iter)
            ## GRASP (ours) ###
            print("Replay Strategy is GRASP")
            data_loader = get_replay_loader(args.images_dir, args.label_dir, 'train', rehearsal_ixs,
              batch_size=args.sleep_batch_size, return_item_ix=True)
            sel_idxs = mnet.grasp_sampling(data_loader, num_iter)

            replay_loader = get_replay_loader(args.images_dir, args.label_dir, 'train', sel_idxs,
              batch_size=args.sleep_batch_size, return_item_ix=False) ## use sleep batch size instead of data loader bs
            print("\nNumber of Samples Selected for Rehearsal:", len(replay_loader.dataset))

            ### Train DNN ###
            mnet.siesta_rehearsal(replay_loader, num_iter)

            test_time_sleep = time.time()
            print("Here is Post-Sleep Evaluations..")
            # All Seen Classes
            probas_all, true_all = mnet.predict(test_loader_all)
            top1_all_dnn, top5_all_dnn = utils.accuracy(probas_all, true_all, topk=(1, 5))
            #print("\nDNN's Top1 accuracy on ALL SEEN classes: %1.2f" % top1_all_dnn)
            print('\nPost Sleep Accuracy [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_all_dnn, top5_all_dnn))
            ## save mnet model
            #mnet.save(max_class, args.save_dir, rehearsal_ixs, class_id_to_item_ix_dict)
            ## OLD Classes
            probas_old, true_old = mnet.predict(test_loader_old)
            top1_old_dnn, top5_old_dnn = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print("DNN's Top1 accuracy on OLD classes: %1.2f" % top1_old_dnn)
            ### NEW Classes
            probas_new, true_new = mnet.predict(test_loader_new)
            top1_new_dnn, top5_new_dnn = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print("DNN's Top1 accuracy on NEW classes: %1.2f" % top1_new_dnn)

            dnn_acc_all = np.append(dnn_acc_all, top1_all_dnn)
            dnn_acc_old = np.append(dnn_acc_old, top1_old_dnn)
            dnn_acc_new = np.append(dnn_acc_new, top1_new_dnn)

            St += time.time() - test_time_sleep
            ##### Reset New Class List #####
            new_class_list = []
            num_recent_stuff = 0
            ####### /// #######
            print('\nTime Spent in Recent Sleep (in MINs): %0.3f' % ((time.time() - start_time_sleep)/60))
            #exit()

        #if max_class == last_class:
        #    break

    exp_dir=args.save_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    np.save(os.path.join(exp_dir, 'val_all_top1.npy'), dnn_acc_all)
    np.save(os.path.join(exp_dir, 'val_old_top1.npy'), dnn_acc_old)
    np.save(os.path.join(exp_dir, 'val_new_top1.npy'), dnn_acc_new)

    #print('\nTotal test time for wake (in mins): %0.3f' % (Wt/60))
    #print('Total test time for pre-sleep (in mins): %0.3f' % (St0/60))
    #print('Total test time for post-sleep (in mins): %0.3f' % (St/60))

    print('\nRuntime for entire experiment (in mins): %0.3f' % ((time.time() - start_time)/60))
    #t = (Wt + St0 + St) / 60
    #tt = ((time.time() - start_time) / 60) - t
    #print("Total run time without test time:", tt)
    ## /// END /// ##

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directories and names
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--label_dir', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir', type=str, default=None)  # directory for ImageNet train/val folders
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
    parser.add_argument('--sleep_samples', type=int, default=25000)  # number of replay samples during sleep

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
    parser.add_argument('--num_classes', type=int, default=1000)  # total number of classes
    parser.add_argument('--min_class', type=int, default=0)  # overall minimum class
    parser.add_argument('--base_init_classes', type=int, default=100)  # number of base init classes
    parser.add_argument('--class_increment', type=int, default=100)  # how often to evaluate
    parser.add_argument('--streaming_min_class', type=int, default=100)  # class to begin stream training
    parser.add_argument('--streaming_max_class', type=int, default=1000)  # class to end stream training
    parser.add_argument('--sleep_batch_size', type=int, default=128)  # training batch size during sleep
    parser.add_argument('--sleep_epoch', type=int, default=50)  # training epoch during sleep
    parser.add_argument('--sleep_lr', type=float, default=0.2)  # starting lr for sleep offline training

    # get arguments and print them out and make any necessary directories
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.lr_mode == 'step_lr_per_class':
        args.lr_gamma = np.exp(args.lr_step_size * np.log(args.end_lr / args.start_lr) / 1300)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # make model and begin stream training
    mnet = MNModel(num_classes=args.num_classes, classifier_G=args.base_arch,
                    extract_features_from=args.extract_features_from, classifier_F=args.classifier,
                    classifier_ckpt=args.classifier_ckpt,
                    weight_decay=args.weight_decay, lr_mode=args.lr_mode, lr_step_size=args.lr_step_size,
                    start_lr=args.start_lr, end_lr=args.end_lr, lr_gamma=args.lr_gamma,
                    num_samples=args.rehearsal_samples, sleep_samples=args.sleep_samples,
                    mixup_alpha=args.mixup_alpha, grad_clip=None,
                    num_channels=args.num_channels, num_feats=args.spatial_feat_dim,
                    num_codebooks=args.num_codebooks, codebook_size=args.codebook_size,
                    max_buffer_size=args.max_buffer_size, sleep_batch_size=args.sleep_batch_size,
                    sleep_epoch=args.sleep_epoch, sleep_lr=args.sleep_lr)

    streaming(args, mnet)
