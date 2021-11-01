#!/bin/bash
python make_hdf5.py --dataset I64 --batch_size 256 --data_root ./data
CUDA_VISIBLE_DEVICES=7 python calculate_inception_moments.py --dataset A128 --data_root ./data