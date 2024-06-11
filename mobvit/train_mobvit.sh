#!/usr/bin/env bash
EXPT_NAME=grasp_rehearsal_experiment_imagenet_300
IMAGE_DIR=/data/datasets/ImageNet2012 # change path to dataset dir
LABEL_ORDER_DIR=./imagenet_files/
BASE_INIT_CKPT=./best_supervised_mobvit_gn_ws_adamw_cosinelr_300e_100c_updated.pth
GPU=0

SLEEP_REPLAY_SAMPLES=1000000 # no need
REPLAY_SAMPLES=64 # no need
MAX_BUFFER_SIZE=130000 # budget=1.51GB(Remind)
CODEBOOK_SIZE=256
NUM_CODEBOOKS=8
BASE_INIT_CLASSES=100
STREAMING_MIN_CLASS=100
CLASS_INCREMENT=25
NUM_CLASSES=1000
EPOCH=1
BATCH=64 # offline training batch
BS=64 # data loader 256
RESUME=supervised_mobvit_gn_ws_adamw_layerlr1e-4_step_MIXUP_CUTMIX_50e_100c

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_exp_mobvit.py \
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
--weight_decay 1e-2 \
--sleep_lr 4e-4 \
--base_arch MobViTClassifyAfterStages3 \
--classifier MobViT_StartAt_Stages3 \
--extract_features_from model.stages3.0 \
--num_channels 128 \
--spatial_feat_dim 16 \
--resume_full_path ${RESUME} \
--save_dir ${EXPT_NAME} \
--ckpt_file ${EXPT_NAME}.pth \
--expt_name ${EXPT_NAME}