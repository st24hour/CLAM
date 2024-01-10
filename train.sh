# GPU 0, 8

##################################### 231211 ######################################
for i in 0.0005
do
    CUDA_VISIBLE_DEVICES=15 python main_clip.py \
                                    --note early_num_layer_0 \
                                    --genomics_encoder 'Genomics_Encoder_FC' \
                                    --num_layers_genom 0 \
                                    --lr $i \
                                    --epochs 1 \
                                    --folds 1 \
                                    --batch_size 64 \
                                    --save_dir '/shared/js.yun/logs/CLAM/clip/base' \
                                    --use_batch
done

##################################### 231206 ######################################
# for i in 0.0005
# do
#     CUDA_VISIBLE_DEVICES=0 python main_clip.py \
#                                     --note early_fusion \
#                                     --genomics_encoder 'Genomics_Encoder_FC_Skip_early_v1' \
#                                     --lr $i \
#                                     --epochs 20 \
#                                     --batch_size 64 \
#                                     --save_dir '/shared/js.yun/logs/CLAM/clip/temp' \
#                                     --use_batch
# done

# # # ##################################### 231128 ######################################
# # random vitb 모델로 뽑은 feature로 clam, LUSC만 따로, clam_mb_multi
# exp_code='TCGA-lung-vitb_256_random'
# # data_root_dir='/shared/j.jang/pathai/data/'
# # feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-x256-features-from-random-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# results_dir='/shared/js.yun/logs/CLAM/Random_feature/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict2='{"LUSC":0, "LUAD":1}'

# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.1
# do
#     CUDA_VISIBLE_DEVICES=15 python main_tmb.py --drop_out \
#                                         --seed 1 \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_mb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --label_dict2 "$label_dict2" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype LUAD LUSC \
#                                         --label_column 'Mutation Count' \
#                                         --loss_balance 0.3 0 0.7 \
#                                         --tmb_high_ratio 0.25
#                                         # --no_inst_cluster
# done

# ##################################### 231127 ######################################
# for i in 0.125 0.25 0.5 1
# do 
#     CUDA_VISIBLE_DEVICES=0 python main_clip.py --epochs 20 --note wd_tuning --wd $i
# done

# ##################################### 231024 ######################################
# # vitb 모델로 뽑은 feature로 clam, LUSC만 따로, clam_mb_multi + label smoothing
# exp_code='TCGA-lung-vitb_256_custom2_big'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict2='{"LUSC":0, "LUAD":1}'

# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.1
# do
#     CUDA_VISIBLE_DEVICES=0 python main_tmb.py --drop_out \
#                                         --seed 1 \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_mb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --label_dict2 "$label_dict2" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype 'LUSC' \
#                                         --label_column 'Mutation Count' \
#                                         --loss_balance 0.3 0 0.7 \
#                                         --label_smoothing $i \
#                                         --tmb_high_ratio 0.6
#                                         # --no_inst_cluster
# done

# ##################################### 231022 ######################################
# # vitb 모델로 뽑은 feature로 clam, LUSC만 따로, clam_mb_multi
# exp_code='TCGA-lung-vitb_256_custom2_big'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# # results_dir='/shared/js.yun/logs/CLAM/temp/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict2='{"LUSC":0, "LUAD":1}'

# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.2 0.25
# do
#     CUDA_VISIBLE_DEVICES=0 python main_tmb.py --drop_out \
#                                         --seed 1 \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_mb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --label_dict2 "$label_dict2" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype 'LUSC' \
#                                         --label_column 'TMB (nonsynonymous)' \
#                                         --loss_balance 0.3 0 0.7 \
#                                         --label_smoothing 0 \
#                                         --tmb_high_ratio $i
#                                         # --no_inst_cluster
# done


# ##################################### 231019 ######################################
# # vitb 모델로 뽑은 feature로 clam, subtype만 돌려서 이거는 제대로 나오는지 test
# exp_code='TCGA-lung-vitb_256_custom2_big'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict2='{"LUSC":0, "LUAD":1}'

# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.25
# do
#     CUDA_VISIBLE_DEVICES=0 python main_tmb.py --drop_out \
#                                         --seed 1 \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_sb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --label_dict2 "$label_dict2" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype 'LUAD' 'LUSC' \
#                                         --label_column 'TMB (nonsynonymous)' \
#                                         --loss_balance 1 0 0 \
#                                         --regression \
#                                         --tmb_high_ratio $i &
#                                         # --no_inst_cluster
#                                         # --focal_loss \
# done
# wait

# ##################################### 231017 ######################################
# # vitb 모델로 뽑은 feature로 clam, LUSC만 따로 
# exp_code='TCGA-lung-vitb_256_custom2_big'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict2='{"LUSC":0, "LUAD":1}'

# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.2 0.25 0.3 0.8
# do
#     CUDA_VISIBLE_DEVICES=0 python main_tmb.py --drop_out \
#                                         --seed 1 \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_sb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --label_dict2 "$label_dict2" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype 'LUSC' \
#                                         --label_column 'Mutation Count' \
#                                         --loss_balance 0.3 0 0.7 \
#                                         --tmb_high_ratio $i &
#                                         # --no_inst_cluster
#                                         # --focal_loss \
# done
# wait

# ##################################### 231010 ######################################
# # HIPT repo pretrained 4k로 우리 데이터 feature 새로 뽑아서 새로운 seed까지 적용하여 CLAM 성능 측정
# # seed 1 
# exp_code='TCGA-lung-vitb_256_custom2_big_230921'

# # data_root_dir='/shared/js.yun/data/CLAM_data/'
# # feature_folder='TCGA-lung-features-DINO-repo-vit-4k/'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
# # label_dict='{"TMB_low":0, "TMB_high":1}'
# label_dict='{"LUSC":0, "LUAD":1}'

# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# # for i in 0.0001 0.00001 0.001 
# for i in 0.2
# do
#     CUDA_VISIBLE_DEVICES=0 python main_tmb.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg 0.00001 \
#                                         --label_smoothing 0 \
#                                         --k 5 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_sb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 \
#                                         --max_epochs 50 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample \
#                                         --target_subtype 'LUSC' 'LUAD' \
#                                         --label_column 'Subtype' \
#                                         --loss_balance 0 1 0 \
#                                         --tmb_high_ratio $i
#                                         # --no_inst_cluster
#                                         # --focal_loss \
# done

# ##################################### 230921 ######################################
# # HIPT repo pretrained 4k로 우리 데이터 feature 새로 뽑아서 새로운 seed까지 적용하여 CLAM 성능 측정
# # seed 1 
# # multi-task로 subtyping 하면서 TMB classification까지
# exp_code='TCGA-lung-luad+lusc-TMB-323-splits_vitb_256_custom2_big_230921-seed1'

# # data_root_dir='/shared/js.yun/data/CLAM_data/'
# # feature_folder='TCGA-lung-features-DINO-repo-vit-4k/'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# # results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# # results_dir='/shared/js.yun/logs/CLAM/HIPT_4k_feature_best_model/'
# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_multi/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits-seed1/task_1_tumor_vs_normal_100/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # subtype까지 고려해서 split 만들었지만 효과 전혀 없음
# # split_dir='/shared/js.yun/data/CLAM_data/test/'

# # for i in 0.0001 0.00001 0.001 
# for i in 0.00001
# do
#     CUDA_VISIBLE_DEVICES=1 python main_multi.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg $i \
#                                         --label_smoothing 0 \
#                                         --k 10 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_sb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 1000 \
#                                         --max_epochs 100 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'custom2_big' \
#                                         --weighted_sample
#                                         # --no_inst_cluster
#                                         # --focal_loss \
# done

# ##################################### 230918 ######################################
# # HIPT repo pretrained 4k로 우리 데이터 feature 새로 뽑아서 새로운 seed까지 적용하여 CLAM 성능 측정
# # seed 1 
# # multi-task로 subtyping 하면서 TMB classification까지
# exp_code='TCGA-lung-luad+lusc-TMB-323-splits-seed1'
# # data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
# # feature_folder='vit256mean_tcga_slide_embeddings/'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-features-DINO-repo-vit-4k/'

# # results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# # results_dir='/shared/js.yun/logs/CLAM/HIPT_4k_feature_best_model/'
# results_dir='/shared/js.yun/logs/CLAM/temp2/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits-seed1/task_1_tumor_vs_normal_100/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # subtype까지 고려해서 split 만들었지만 효과 전혀 없음
# # split_dir='/shared/js.yun/data/CLAM_data/test/'

# # for i in 0.0001 0.00001 0.001 
# for i in 0.0001
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg $i \
#                                         --label_smoothing 0 \
#                                         --k 10 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task multi_task \
#                                         --model_type clam_sb_multi \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 1000 \
#                                         --max_epochs 200 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'HIPT_4k_feat' \
#                                         --weighted_sample \
#                                         --no_inst_cluster
#                                         # --focal_loss \
# done

# ##################################### 230918 ######################################
# # HIPT repo pretrained 4k로 우리 데이터 feature 새로 뽑아서 새로운 seed까지 적용하여 CLAM 성능 측정
# # seed 1 
# exp_code='TCGA-lung-luad+lusc-TMB-323-splits-seed1'
# # data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
# # feature_folder='vit256mean_tcga_slide_embeddings/'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-features-DINO-repo-vit-4k/'

# # results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# results_dir='/shared/js.yun/logs/CLAM/HIPT_4k_feature_best_model/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits-seed1/task_1_tumor_vs_normal_100/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # subtype까지 고려해서 split 만들었지만 효과 전혀 없음
# # split_dir='/shared/js.yun/data/CLAM_data/test/'

# for i in 0.0001 0.00001 0.001 
# do
#     CUDA_VISIBLE_DEVICES=6 python main.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg $i \
#                                         --label_smoothing 0 \
#                                         --k 10 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task task_1_tumor_vs_normal \
#                                         --model_type clam_mb \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 1000 \
#                                         --max_epochs 200 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'HIPT_4k_feat' \
#                                         --weighted_sample \
#                                         --no_inst_cluster
#                                         # --focal_loss \
# done

# ##################################### 230907 ######################################
# # HIPT repo pretrained 4k feature로 CLAM 성능 측정
# # label smoothing 추가
# exp_code='TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only'
# data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
# feature_folder='vit256mean_tcga_slide_embeddings/'
# # results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# results_dir='/shared/js.yun/logs/CLAM/HIPT_4k_feature_best_model/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # 밑에 2개 처리 해야됨
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'

# # subtype까지 고려해서 split 만들었지만 효과 전혀 없음
# # split_dir='/shared/js.yun/data/CLAM_data/test/'

# # for i in 0.0001 0.00001 0.001 
# # do
# #     CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
# #                                         --lr 2e-4 \
# #                                         --reg $i \
# #                                         --label_smoothing 0 \
# #                                         --focal_loss \
# #                                         --k 1 \
# #                                         --label_frac 1 \
# #                                         --exp_code $exp_code \
# #                                         --bag_loss ce \
# #                                         --inst_loss svm \
# #                                         --task task_1_tumor_vs_normal \
# #                                         --model_type clam_mb \
# #                                         --log_data \
# #                                         --data_root_dir $data_root_dir \
# #                                         --feature_folder $feature_folder \
# #                                         --results_dir $results_dir \
# #                                         --split_dir $split_dir \
# #                                         --csv_path $csv_path \
# #                                         --subtyping \
# #                                         --opt adam \
# #                                         --decay_epoch 300 1000 \
# #                                         --max_epochs 200 \
# #                                         --label_dict "$label_dict" \
# #                                         --model_size 'HIPT_4k_feat' \
# #                                         --weighted_sample \
# #                                         --no_inst_cluster
# # done

# # for i in 0.0001 0.00001 0.001 
# # do
# #     CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
# #                                         --lr 2e-4 \
# #                                         --reg $i \
# #                                         --label_smoothing 0 \
# #                                         --focal_loss \
# #                                         --k 1 \
# #                                         --label_frac 1 \
# #                                         --exp_code $exp_code \
# #                                         --bag_loss ce \
# #                                         --inst_loss svm \
# #                                         --task task_1_tumor_vs_normal \
# #                                         --model_type clam_mb \
# #                                         --log_data \
# #                                         --data_root_dir $data_root_dir \
# #                                         --feature_folder $feature_folder \
# #                                         --results_dir $results_dir \
# #                                         --split_dir $split_dir \
# #                                         --csv_path $csv_path \
# #                                         --subtyping \
# #                                         --opt adamw \
# #                                         --decay_epoch 300 1000 \
# #                                         --max_epochs 200 \
# #                                         --label_dict "$label_dict" \
# #                                         --model_size 'HIPT_4k_feat' \
# #                                         --weighted_sample \
# #                                         --no_inst_cluster
# # done

# for i in 0.0001 0.00001 0.001 
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg $i \
#                                         --label_smoothing 0 \
#                                         --focal_loss \
#                                         --k 1 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task task_1_tumor_vs_normal \
#                                         --model_type clam_mb \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adamw \
#                                         --decay_epoch 300 1000 \
#                                         --max_epochs 200 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'HIPT_4k_feat' \
#                                         --no_inst_cluster
#                                         # --weighted_sample \
# done

##################################### 230806 ######################################
# # DINO_ours vit256으로 뽑은 feature로 CLAM 돌림
# exp_code='TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_DINO_ours-vit256'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-features-DINO-ours-vit-256-epoch4'
# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vit256_epoch4_feature/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# # 밑에 2개 처리 해야됨
# split_dir='/shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# for i in 0.00001 0.0001 0.001
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                         --lr 2e-4 \
#                                         --reg $i \
#                                         --k 1 \
#                                         --label_frac 1 \
#                                         --exp_code $exp_code \
#                                         --weighted_sample \
#                                         --bag_loss ce \
#                                         --inst_loss svm \
#                                         --task task_1_tumor_vs_normal \
#                                         --model_type clam_mb \
#                                         --log_data \
#                                         --data_root_dir $data_root_dir \
#                                         --feature_folder $feature_folder \
#                                         --results_dir $results_dir \
#                                         --split_dir $split_dir \
#                                         --csv_path $csv_path \
#                                         --subtyping \
#                                         --opt adam \
#                                         --decay_epoch 300 1000 \
#                                         --max_epochs 30 \
#                                         --label_dict "$label_dict" \
#                                         --model_size 'HIPT_256_feat'
#                                         # --no_inst_cluster
# done

##################################### 230801 ######################################
# # # HIPT feature가 4k만 있고 256 feature가 없어서 올려준 vit256 model로 직접 feature 뽑음
# # 중간에 계속 멈춰서 0,1번 GPU로 다시 돌림
# #### extract_features_fp.py
# # data_h5_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/'
# # data_slide_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney/'
# # csv_path='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/process_list_autogen.csv'
# # feat_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-features/'

# data_h5_dir='/shared/j.jang/pathai/data/TCGA-lung-patches-256'
# data_slide_dir='/shared/j.jang/pathai/data/TCGA-lung/'
# csv_path='/shared/j.jang/pathai/data/TCGA-lung-patches-256/process_list_autogen.csv'
# feat_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-features-DINO-ours-vit-256/'
# model_256_path='/shared/j.jang/pathai/HIPT/dino_ckpt_vit-s-256_10epoch_lung/checkpoint.pth'
# # model_256_path='/shared/js.yun/HIPT/HIPT_original/HIPT_4K/Checkpoints/vit256_small_dino.pth'

# CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir $data_h5_dir \
#                                                         --data_slide_dir $data_slide_dir \
#                                                         --csv_path $csv_path \
#                                                         --feat_dir $feat_dir \
#                                                         --model_256_path $model_256_path \
#                                                         --image_encoder 'HIPT_256' \
#                                                         --batch_size 512 \
#                                                         --loader 'default' \
#                                                         --num_worker 8

##################################### 230803 ######################################
# # HIPT pretrained vit256으로 뽑은 feature로 CLAM 돌리려고 했는데 실수로 feature 저장을 안함...
# # 8월 3일에 돌릴 예정
# exp_code='TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_pretrained-vit256'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-features-HIPT-repo-pretrained-256'
# results_dir='/shared/js.yun/logs/CLAM/HIPT_256_feature/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# # 밑에 2개 처리 해야됨
# split_dir='/shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                     --lr 2e-4 \
#                                     --reg 0.00001 \
#                                     --k 1 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_1_tumor_vs_normal \
#                                     --model_type clam_mb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path \
#                                     --subtyping \
#                                     --opt adam \
#                                     --decay_epoch 300 1000 \
#                                     --max_epochs 30 \
#                                     --label_dict "$label_dict" \
#                                     --model_size 'HIPT_256_feat'
#                                     # --no_inst_cluster

##################################### 230802 ######################################
# # # # HIPT feature가 4k만 있고 256 feature가 없어서 올려준 vit256 model로 직접 feature 뽑음
# # # 해보니까 worker8 일떄 batch_size 4096보다 2048이 더 빠름
# #### extract_features_fp.py
# # data_h5_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/'
# # data_slide_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney/'
# # csv_path='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/process_list_autogen.csv'
# # feat_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-features/'

# data_h5_dir='/shared/j.jang/pathai/data/TCGA-lung-patches-256'
# data_slide_dir='/shared/j.jang/pathai/data/TCGA-lung/'
# csv_path='/shared/j.jang/pathai/data/TCGA-lung-patches-256/process_list_autogen.csv'
# feat_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-features-HIPT-repo-pretrained-256/'
# model_256_path='/shared/js.yun/HIPT/HIPT_original/HIPT_4K/Checkpoints/vit256_small_dino.pth'

# CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir $data_h5_dir \
#                                                         --data_slide_dir $data_slide_dir \
#                                                         --csv_path $csv_path \
#                                                         --feat_dir $feat_dir \
#                                                         --model_256_path $model_256_path \
#                                                         --image_encoder 'HIPT_256' \
#                                                         --batch_size 512 \
#                                                         --loader 'default' \
#                                                         --num_worker 8

##################################### 230726 ######################################
# exp_code='TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only'
# data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
# feature_folder='vit256mean_tcga_slide_embeddings/'
# results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # 밑에 2개 처리 해야됨
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'

# CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                     --lr 2e-4 \
#                                     --k 1 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_1_tumor_vs_normal \
#                                     --model_type clam_mb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path \
#                                     --subtyping \
#                                     --opt adam \
#                                     --decay_epoch 300 1000 \
#                                     --max_epochs 200 \
#                                     --label_dict "$label_dict" \
#                                     --model_size 'HIPT_feat' \
#                                     --no_inst_cluster


##################################### 230725 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
# feature_folder='vit256mean_tcga_slide_embeddings/'
# results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # 밑에 2개 처리 해야됨
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'

# CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
#                                     --lr 2e-4 \
#                                     --k 1 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_1_tumor_vs_normal \
#                                     --model_type clam_sb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path \
#                                     --subtyping \
#                                     --opt adam \
#                                     --decay_epoch 300 1000 \
#                                     --max_epochs 200 \
#                                     --label_dict "$label_dict" \
#                                     --model_size 'HIPT_feat' \
#                                     --no_inst_cluster



# ##### create_splits_seq.py
# # HIPT에서 뽑은 feature가 있는 데이터만 가지고 split 만듦
# # 종성님이 task1으로 split 하셔서 그대로 task1으로 따라감
# # csv 파일은 gen_dataset_csv_from_clinical_data.ipynb 으로 만듦
# # csv 파일에 case_id 넣어도 되나? -> csv은 상관 없음. 나중에 무시하면 됨
# # case_id는 안쓰도록 코드 수정해야됨
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/'
# label_dict='{"TMB_low":0, "TMB_high":1}'
# python create_splits_seq.py --task task_1_tumor_vs_normal \
#                             --seed 1 \
#                             --label_frac 1.0 \
#                             --val_frac 0.1 \
#                             --test_frac 0.2 \
#                             --k 1 \
#                             --n_class 2 \
#                             --csv_path $csv_path \
#                             --split_dir $split_dir \
#                             --label_dict "$label_dict"

# # 종성님 creat_split.sh 에서 가져옴
# python create_splits_seq.py \
# --task task_1_tumor_vs_normal \
# --seed 1 \
# --label_frac 1.0 \
# --val_frac 0.1 \
# --test_frac 0.2 \
# --k 1 \
# --label_dict "{'TMB_low':0, 'TMB_high':1}" \
# --csv_path /shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv \
# --split_dir /shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits/

##################################### 230508 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# # results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out \
#                                     --early_stopping \
#                                     --lr 2e-4 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type clam_sb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path \
#                                     --subtyping \
#                                     --opt adam \
#                                     --decay_epoch 300 1000 \
#                                     --max_epochs 200 \
#                                     --label_dict "$label_dict"



# CUDA_VISIBLE_DEVICES=10 python create_patches_fp.py --patch_size 256 --seg --patch --stitch 



##### extract_features_fp.py
# data_h5_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/'
# data_slide_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney/'
# csv_path='/shared/js.yun/data/CLAM_data/TCGA-kidney-patches/process_list_autogen.csv'
# feat_dir='/shared/js.yun/data/CLAM_data/TCGA-kidney-features/'

# CUDA_VISIBLE_DEVICES=8,9 python extract_features_fp.py --data_h5_dir $data_h5_dir \
#                                                         --data_slide_dir $data_slide_dir \
#                                                         --csv_path $csv_path \
#                                                         --feat_dir $feat_dir



##### create_splits_seq.py
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/'
# python create_splits_seq.py --task task_1_tumor_vs_normal \
#                             --seed 1 \
#                             --label_frac 1 \
#                             --k 10 \
#                             --n_class 2 \
#                             --csv_path $csv_path \
#                             --split_dir $split_dir



##### 
# exp_code='LUSC_vs_LUAD_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_1_tumor_vs_normal_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'

# CUDA_VISIBLE_DEVICES=10 python main.py --drop_out \
#                                     --early_stopping \
#                                     --lr 2e-4 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_1_tumor_vs_normal \
#                                     --model_type clam_sb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path


##### 
# exp_code='sgd0.01_LUSC_vs_LUAD_CLAM_100_task2_mb'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out \
#                                     --early_stopping \
#                                     --lr 0.01 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type clam_mb \
#                                     --log_data \
#                                     --data_root_dir $data_root_dir \
#                                     --feature_folder $feature_folder \
#                                     --results_dir $results_dir \
#                                     --split_dir $split_dir \
#                                     --csv_path $csv_path \
#                                     --subtyping \
#                                     --opt sgd \
#                                     --decay_epoch 30 50