#!/usr/bin/env bash
EXPT_NAME=grasp_rehearsal_places_exp
IMAGE_DIR=/data/datasets/Places_LT # change path to dataset dir
LABEL_ORDER_DIR=./places_indices
BASE_INIT_CKPT=/GRASP/SIESTA/latent/swav_100c_2000e_mobilenet_modified_gelu_updated.pth
GPU=0

REPLAY_SAMPLES=32 # no need
MAX_BUFFER_SIZE=20000 # memory constraints
CODEBOOK_SIZE=256
NUM_CODEBOOKS=8
BASE_INIT_CLASSES=65 # no need
STREAMING_MIN_CLASS=0
CLASS_INCREMENT=73
NUM_CLASSES=365
EPOCH=40
BATCH=32 # offline training batch size 32 during rehearsal cycle
BS=256 # data loader 256
SEED=1 #1, 74, 1444, 1883, 1993, 2023
RESUME=/GRASP/SIESTA/cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c


CUDA_VISIBLE_DEVICES=${GPU} python -u places_LT_exp.py \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${STREAMING_MIN_CLASS} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier_ckpt ${BASE_INIT_CKPT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--sleep_epoch ${EPOCH} \
--sleep_batch_size ${BATCH} \
--batch_size ${BS} \
--weight_decay 1e-5 \
--sleep_lr 0.1 \
--seed ${SEED} \
--base_arch MobNetClassifyAfterLayer8 \
--classifier MobNet_StartAt_Layer8 \
--extract_features_from model.features.7 \
--num_channels 80 \
--spatial_feat_dim 14 \
--resume_full_path ${RESUME} \
--save_dir ${EXPT_NAME} \
--ckpt_file ${EXPT_NAME}.pth \
--expt_name ${EXPT_NAME}
