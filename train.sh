# GPU 0, 8

##################################### 230725 ######################################
# 

exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
data_root_dir='/shared/js.yun/HIPT/HIPT_original/3-Self-Supervised-Eval/embeddings_slide_lib/embeddings_slide_lib/'
feature_folder='vit256mean_tcga_slide_embeddings/'
results_dir='/shared/js.yun/logs/CLAM/HIPT_feature/'

# 밑에 2개 처리 해야됨
split_dir /shared/j.jang/pathai/data/TCGA-lung-luad+lusc-TMB-323-splits/task_1_tumor_vs_normal_100/
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'

label_dict='{"LUSC":0, "LUAD":1}'

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out \
                                    --lr 2e-4 \
                                    --k 1 \
                                    --label_frac 1 \
                                    --exp_code $exp_code \
                                    --weighted_sample \
                                    --bag_loss ce \
                                    --inst_loss svm \
                                    --task task_2_tumor_subtyping \
                                    --model_type clam_sb \
                                    --no_inst_cluster \
                                    --log_data \
                                    --data_root_dir $data_root_dir \
                                    --feature_folder $feature_folder \
                                    --results_dir $results_dir \
                                    --split_dir $split_dir \
                                    --csv_path $csv_path \
                                    --subtyping \
                                    --opt adam \
                                    --decay_epoch 300 1000 \
                                    --max_epochs 200 \
                                    --label_dict "$label_dict" \
                                    --batch_size 16 \
                                    --num_workers 16 


##################################### 230529 ######################################
# ##### create_splits_seq.py
# # HIPT에서 사용한 lung slide만 가지고 csv 파일을 만들었기 떄문에 그거만 가지고 다시 split 만듦
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA_lung_HIPT_only.csv'
# split_dir='/shared/js.yun/data/CLAM_data/temp/'
# python create_splits_seq.py --task task_2_tumor_subtyping \
#                             --seed 1 \
#                             --label_frac 1 \
#                             --k 10 \
#                             --n_class 2 \
#                             --csv_path $csv_path \
#                             --split_dir $split_dir


##################################### 230704 ######################################
# # TCGA-colorectal-patches 만드는 코드
# python create_patches_fp.py --patch_size 256 --seg --patch --stitch \
#                             --save_dir '/shared/js.yun/data/CLAM_data/TCGA-colorectal-patches/' \
#                             --source '/shared/j.jang/pathai/data/TCGA-colorectum'

##################################### 230702 ######################################
# TCGA-colorectal-splits 만드는 코드
# label_dict='{"MSS":0, "MSI":1}'
# python create_splits_seq.py \
# --task task_2_tumor_subtyping \
# --seed 1 \
# --label_frac 1.0 \
# --val_frac 0.1 \
# --test_frac 0.2 \
# --k 1 \
# --label_dict "$label_dict" \
# --csv_path /shared/js.yun/CLAM/dataset_csv/TCGA-colorectal.csv \
# --split_dir /shared/js.yun/data/CLAM_data/TCGA-colorectal-splits/
# # --split_dir /shared/j.jang/pathai/data/TCGA-colorectal-adenocarcinoma-splits/

##################################### 230531 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# # results_dir='/shared/js.yun/logs/CLAM/temp/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out 0 \
#                                     --lr 0.001 \
#                                     --reg 0.0001 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type mlp_mixer_s0 \
#                                     --no_inst_cluster \
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
#                                     --num_patch 5000 \
#                                     --batch_size 16 \
#                                     --num_workers 16 \
#                                     --dim 4096 \
#                                     --depth 5 \
#                                     --expansion_factor_patch 0.25 \
#                                     --expansion_factor 1

##################################### 230529 ######################################
##### create_splits_seq.py
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'
# split_dir='/shared/js.yun/data/CLAM_data/temp/'
# python create_splits_seq.py --task task_1_tumor_vs_normal \
#                             --seed 1 \
#                             --label_frac 1 \
#                             --k 10 \
#                             --n_class 2 \
#                             --csv_path $csv_path \
#                             --split_dir $split_dir


##################################### 230526 ######################################
# CUDA_VISIBLE_DEVICES=8 python create_patches_fp.py --patch_size 256 --seg --patch --stitch \
#                                 --save_dir '/shared/js.yun/data/CLAM_data/temp/' \
# 	                            --source '/shared/js.yun/data/CLAM_data/TCGA-kidney/'


##################################### 230525 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# # results_dir='/shared/js.yun/logs/CLAM/temp/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out 0.5 \
#                                     --lr 0.001 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type mlp_mixer_s0 \
#                                     --no_inst_cluster \
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
#                                     --num_patch 5000 \
#                                     --batch_size 16 \
#                                     --num_workers 16 \
#                                     --dim 256 \
#                                     --depth 10 \
#                                     --expansion_factor_patch 0.1 \
#                                     --expansion_factor 4


##################################### 230516 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# # results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# results_dir='/shared/js.yun/logs/CLAM/temp/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out \
#                                     --lr 2e-4 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type clam_sb \
#                                     --no_inst_cluster \
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
#                                     --num_patch 3000 \
#                                     --batch_size 16 \
#                                     --num_workers 16 


##################################### 230516 ######################################
# exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
# data_root_dir='/shared/js.yun/data/CLAM_data/'
# feature_folder='TCGA-breast-features'
# results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
# # results_dir='/shared/js.yun/logs/CLAM/temp/'
# split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
# csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
# label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

# CUDA_VISIBLE_DEVICES=8 python main.py --drop_out \
#                                     --lr 2e-4 \
#                                     --k 10 \
#                                     --label_frac 1 \
#                                     --exp_code $exp_code \
#                                     --weighted_sample \
#                                     --bag_loss ce \
#                                     --inst_loss svm \
#                                     --task task_2_tumor_subtyping \
#                                     --model_type clam_sb \
#                                     --no_inst_cluster \
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
#                                     --num_patch 3000 \
#                                     --batch_size 16 \
#                                     --num_workers 16 


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
#                                     --no_inst_cluster \
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