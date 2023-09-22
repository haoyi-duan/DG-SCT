python net_grd_avst/main_avst.py --mode train \
	--audio_dir data/AVQA/vggish \
	--video_res14x14_dir data/AVQA/frames/ \
	--wandb 0 --num_workers 16 --batch-size 8 --model_name swinv2_tune_av+vggish --backbone_type audioset --Adapter_downsample 8 --num_tokens 2