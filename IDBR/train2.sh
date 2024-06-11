python -u ./src/naivereplay.py \
--gpu 0 \
--seed 0 \
--tasks dbpedia yahoo ag amazon yelp \
--epochs 1 1 1 1 1 \
--batch_size 8 \
--replay_freq 1 \
--store_ratio 0.5
