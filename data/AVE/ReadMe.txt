AVE.zip: videos in AVE dataset. One video may contain different audio-visual events, so the total number of videos is not 4143.

annotations.txt: annotations of AVE dataset. For each sample, you can find its event catergory, YouTube ID, 
                 Quality (all good, means that it contains an AVE), start time of an audio-visual event, end 
				 time of an audio-visual event.

train/val/test-Set.txt: training/validation/testing set used in our ECCV paper.
 

If you used our AVE dataset, please consider cite our paper:
@inproceedings{TianECCV2018,
	 title={Audio-Visual Event Localization in Unconstrained Videos},
     author={Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
	 editor="Ferrari, Vittorio and Hebert, Martial and Sminchisescu, Cristian and Weiss, Yair",
	 booktitle="Computer Vision -- ECCV 2018",
	 year="2018",
	 publisher="Springer",
}