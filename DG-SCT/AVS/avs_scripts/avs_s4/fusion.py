import os
import cv2
import pandas as pd
import pdb

def save_raw_img_mask(anno_file_path, raw_img_base_path, mask_base_path, split='test', r=0.5):
    df = pd.read_csv(anno_file_path, sep=',')
    df_test = df[df['split'] == split]
    count = 0
    for video_id in range(len(df_test)):
        video_name, category = df_test.iloc[video_id][0], df_test.iloc[video_id][2]
        raw_img_path = os.path.join(raw_img_base_path, split, category, video_name)
        for img_id in range(5):
            img_name = "%s_%d.png"%(video_name, img_id + 1)
            raw_img = cv2.imread(os.path.join(raw_img_path, img_name))
            mask = cv2.imread(os.path.join(mask_base_path, category, video_name, "%s_%d.png"%(video_name, img_id)))    
            #pdb.set_trace()
            mask=cv2.resize(mask,(raw_img.shape[1],raw_img.shape[0]),interpolation=cv2.INTER_AREA)
            raw_img_mask = cv2.addWeighted(raw_img, 1, mask, r, 0)
            save_img_path = os.path.join(mask_base_path, 'img_add_masks', category, video_name)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_img_path, img_name), raw_img_mask)
        count += 1
        
    print(f'count: {count} videos')

anno_file_path="/data/AVSBench_data/Single-source/s4_meta_data.csv"
raw_img_base_path="/data/AVSBench_data/Single-source/s4_data/visual_frames"
mask_base_path="/LAVISH/AVS/avs_scripts/avs_s4/mask"
save_raw_img_mask(anno_file_path,raw_img_base_path,mask_base_path)