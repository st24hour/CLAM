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
exp_code='LUSC_vs_LUAD_CLAM_100_task2_mb'
data_root_dir='/shared/js.yun/data/CLAM_data/'
feature_folder='TCGA-lung-features'
results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_2_tumor_subtyping_100/'
csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'

CUDA_VISIBLE_DEVICES=11 python main.py --drop_out \
                                    --early_stopping \
                                    --lr 2e-4 \
                                    --k 10 \
                                    --label_frac 1 \
                                    --exp_code $exp_code \
                                    --weighted_sample \
                                    --bag_loss ce \
                                    --inst_loss svm \
                                    --task task_2_tumor_subtyping \
                                    --model_type clam_mb \
                                    --log_data \
                                    --data_root_dir $data_root_dir \
                                    --feature_folder $feature_folder \
                                    --results_dir $results_dir \
                                    --split_dir $split_dir \
                                    --csv_path $csv_path \
                                    --subtyping
