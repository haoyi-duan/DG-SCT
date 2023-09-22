
setting='S4'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "./train_logs/S4_pvt_20230511-172418/checkpoints/S4_pvt_best.pth" \
    --num_tokens 32 --Adapter_downsample 8 \
    --test_batch_size 4 \
    --tpavi_va_flag 1 \
    --tpavi_stages 0 1 2 3
    --save_pred_mask \