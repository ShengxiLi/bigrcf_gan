# use z_var to change the variance of z for all the sampling
# use --mybn --accumulate_stats --num_standing_accumulations 32 to 
# use running stats
CUDA_VISIBLE_DEVICES=7 python sample.py \
--dataset L64 --parallel --shuffle  --num_workers 8 --batch_size 64  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 5 --G_lr 2e-4 --D_lr 2e-4 --D_B2 0.900 --G_B2 0.900 \
--G_attn 0 --D_attn 0 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--D_thin \
--dim_z 128 --shared_dim 1 \
--G_init xavier --D_init xavier \
--G_eval_mode \
--name_suffix SNGAN \
--data_root '/mnt/disk1/lsx/gan_dataset/' \
--unconditional \
--which_best FID \
--num_epochs 1000 \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--skip_init --load_weights best0 \
--sample_inception_metrics  --sample_random --sample_sheets --sample_interps \