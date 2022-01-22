#!/bin/bash
#python make_hdf5.py --dataset C10 --batch_size 256 --data_root ./data
CUDA_VISIBLE_DEVICES=0 python calculate_inception_moments.py --dataset C10 --data_root ./data