# GPU 4, 12

##################################### 231022 ######################################
# vitb 모델로 뽑은 feature로 clam, LUSC만 따로, clam_mb_multi
exp_code='TCGA-lung-vitb_256_custom2_big'
data_root_dir='/shared/j.jang/pathai/data/'
feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_231017/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
label_dict='{"TMB_low":0, "TMB_high":1}'
label_dict2='{"LUSC":0, "LUAD":1}'

split_dir='/shared/js.yun/data/CLAM_data/'
csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# label_column 'TMB (nonsynonymous)', 'Mutation Count', 'Subtype'
# for i in 0.0001 0.00001 0.001 
for i in 0.6 0.65
do
    CUDA_VISIBLE_DEVICES=4 python main_tmb.py --drop_out \
                                        --seed 1 \
                                        --lr 2e-4 \
                                        --reg 0.00001 \
                                        --label_smoothing 0 \
                                        --k 5 \
                                        --label_frac 1 \
                                        --exp_code $exp_code \
                                        --bag_loss ce \
                                        --inst_loss svm \
                                        --task multi_task \
                                        --model_type clam_mb_multi \
                                        --log_data \
                                        --data_root_dir $data_root_dir \
                                        --feature_folder $feature_folder \
                                        --results_dir $results_dir \
                                        --split_dir $split_dir \
                                        --csv_path $csv_path \
                                        --subtyping \
                                        --opt adam \
                                        --decay_epoch 300 \
                                        --max_epochs 50 \
                                        --label_dict "$label_dict" \
                                        --label_dict2 "$label_dict2" \
                                        --model_size 'custom2_big' \
                                        --weighted_sample \
                                        --target_subtype 'LUSC' \
                                        --label_column 'TMB (nonsynonymous)' \
                                        --loss_balance 0.3 0 0.7 \
                                        --tmb_high_ratio $i 
                                        # --no_inst_cluster
                                        # --focal_loss \
done


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
# for i in 0.2 0.25 0.3 0.35
# do 
#     CUDA_VISIBLE_DEVICES=4 python main_tmb.py --drop_out \
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
#                                         --label_column 'TMB (nonsynonymous)' \
#                                         --loss_balance 0.3 0 0.7 \
#                                         --tmb_high_ratio $i &
#                                         # --no_inst_cluster
#                                         # --focal_loss \
# done
# wait

# ##################################### 231010 ######################################
# # HIPT repo pretrained 4k로 우리 데이터 feature 새로 뽑아서 새로운 seed까지 적용하여 CLAM 성능 측정
# # seed 1 
# exp_code='TCGA-lung-luad+lusc-TMB-323-splits_vitb_256_custom2_big_230921'

# # data_root_dir='/shared/js.yun/data/CLAM_data/'
# # feature_folder='TCGA-lung-features-DINO-repo-vit-4k/'
# data_root_dir='/shared/j.jang/pathai/data/'
# feature_folder='TCGA-lung-x256-features-dino-from-pretrained-vitb-img224/'

# results_dir='/shared/js.yun/logs/CLAM/DINO_ours_vitb_256_custom2_big_230921/'
# label_dict='{"TMB_low":0, "TMB_high":1}'

# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-luad+lusc-TMB-323-HIPT-only-splits/task_1_tumor_vs_normal_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323_HIPT_only.csv'
# split_dir='/shared/js.yun/data/CLAM_data/'
# csv_path='/shared/j.jang/pathai/CLAM/dataset_csv/TCGA-lung-LUAD+LUSC-TMB-pan_cancer-323.csv'

# # for i in 0.0001 0.00001 0.001 
# for i in 0.4
# do
#     CUDA_VISIBLE_DEVICES=4 python main_tmb.py --drop_out \
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
#                                         --target_subtype 'LUSC' \
#                                         --label_column 'TMB (nonsynonymous)' \
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

# ##################################### 230430 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=12 python main.py --drop_out \
#                                     --lr 2e-4 \
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
#                                     --opt adam \
#                                     --decay_epoch 100 150 \
#                                     --max_epochs 200 \
#                                     --label_dict "$label_dict" \
#                                     --attn 'js'



# exp_code='js_LUSC_vs_LUAD_CLAM_100_task2_mb'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-lung-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'
# label_dict='{"LUSC":0,"LUAD":1,"asdf":232323}'
# # exp_code='js_LUSC_vs_LUAD_CLAM_100_task2_mb'
# # data_root_dir='/shared/js.yun/data/CLAM_data/'
# # feature_folder='TCGA-lung-features'
# # results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
# # split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_2_tumor_subtyping_100/'
# # csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'
# # label_dict='{"LUSC":0,"LUAD":1}'

# CUDA_VISIBLE_DEVICES=12 python main.py --drop_out \
#                                     --early_stopping \
#                                     --lr 2e-4 \
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
#                                     --gate 'js' \
#                                     --label_dict $label_dict
                                    
