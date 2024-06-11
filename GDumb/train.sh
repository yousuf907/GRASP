#!/usr/bin/env bash
EXPT_NAME=GDumb_veridical_rehearsal_imagenet_1k_exp
IMAGE_DIR=/data/datasets/ImageNet1K # change path to dataset dir
LABEL_ORDER_DIR=./imagenet_files/
BASE_INIT_CKPT=GRASP/SIESTA/veridical/base_init_ER_mobilenet_swav_cosine_loss_sgd_step_aug_100c_50e_updated.pth
GPU=0

SLEEP_REPLAY_SAMPLES=1000000 # not needed
REPLAY_SAMPLES=256 # not needed
MAX_BUFFER_SIZE=130000 # memory constraints
CODEBOOK_SIZE=256 # not needed
NUM_CODEBOOKS=8 # not needed
BASE_INIT_CLASSES=100
STREAMING_MIN_CLASS=100
CLASS_INCREMENT=50
NUM_CLASSES=1000
EPOCH=50
BATCH=256 # training mini-batch size during rehearsal
BS=256 # data loader 256

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_exp_gdumb.py \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${STREAMING_MIN_CLASS} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier_ckpt ${BASE_INIT_CKPT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--sleep_samples ${SLEEP_REPLAY_SAMPLES} \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--sleep_epoch ${EPOCH} \
--sleep_batch_size ${BATCH} \
--batch_size ${BS} \
--weight_decay 1e-5 \
--sleep_lr 0.4 \
--base_arch MobNetClassifyAfterLayer8 \
--classifier MobNet_StartAt_Layer8 \
--extract_features_from model.features.7 \
--num_channels 80 \
--spatial_feat_dim 14 \
--save_dir ${EXPT_NAME} \
--ckpt_file ${EXPT_NAME}.pth \
--expt_name ${EXPT_NAME}
