setting='MS3'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"

# python test.py \
#     --session_name ${setting}_${visual_backbone} \
#     --visual_backbone ${visual_backbone} \
#     --weights "./train_logs/MS3_pvt_20230515-002825/checkpoints/MS3_pvt_best.pth" \
#     --test_batch_size 4 \
#     --tpavi_stages 0 1 2 3 \
#     --tpavi_va_flag  1 \
#     --save_pred_mask \

python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "./train_logs/MS3_pvt_20230513-230931/checkpoints/MS3_pvt_best.pth" \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag  1 \
    --gamma 0.05 \
    --save_pred_mask \

