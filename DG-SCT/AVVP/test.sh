python main.py --mode test \
    --model_save_dir models/ \
    --unimodal_assign soft --crossmodal_assign soft \
    --Adapter_downsample 8 --batch_size 8 \
    --is_audio_adapter_p1 1 --is_audio_adapter_p2 1 --is_audio_adapter_p3 0 \
    --is_before_layernorm 1 --is_bn 1 --is_fusion_before 1 --is_gate 1  --is_post_layernorm 1 --is_vit_ln 0 \
    --num_conv_group 2 --num_tokens 32 --num_workers 16