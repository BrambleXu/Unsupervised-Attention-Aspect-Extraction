#!/usr/bin/env bash
# THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
# --emb ../preprocessed_data/restaurant/w2v_embedding \
#--domain restaurant \
# -o output_dir \
# python3 train.py --emb ../preprocessed_data/restaurant/w2v_embedding --domain restaurant -o output_dir

# python train.py --emb ../preprocessed_data/restaurant/w2v_embedding --domain restaurant -o output_dir


THEANO_FLAGS="device=cuda0,floatX=float32" python train.py \
--emb ../preprocessed_data/$domain/w2v_embedding \
--domain restaurant \
-o output_dir \