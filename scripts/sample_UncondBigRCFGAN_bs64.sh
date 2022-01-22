# use z_var to change the variance of z for all the sampling
# use --mybn --accumulate_stats --num_standing_accumulations 32 to 
# use running stats
CUDA_VISIBLE_DEVICES=6 python sample.py \
--model RcfGAN --which_train_fn RCFGAN --t_sigma_num 64 \
--z_var 0.5 \
--dataset A128 --parallel --shuffle  --num_workers 8 --batch_size 64  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_ch 64 --D_ch 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho --skip_init \
--hier --dim_z 128 --shared_dim 1 \
--ema --ema_start 20000 \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--skip_init --use_ema --G_eval_mode --load_weights best0 \
--sample_inception_metrics --sample_random --sample_sheets --sample_interps \
--unconditional \
--name_suffix reg6 \
--data_root './data' \