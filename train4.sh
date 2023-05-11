# GPU 4, 12

##################################### 230430 ######################################
exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
data_root_dir='/shared/js.yun/data/CLAM_data/'
feature_folder='TCGA-breast-features'
results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

CUDA_VISIBLE_DEVICES=12 python main.py --drop_out \
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
                                    --subtyping \
                                    --opt adam \
                                    --decay_epoch 100 150 \
                                    --max_epochs 200 \
                                    --label_dict "$label_dict" \
                                    --attn 'js'



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
                                    
