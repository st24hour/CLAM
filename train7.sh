# GPU 7, 15

##################################### 230430 ######################################
exp_code='LUSC_vs_LUAD_CLAM_100_task2_mb'
data_root_dir='/shared/js.yun/data/CLAM_data/'
feature_folder='TCGA-lung-features'
results_dir='/shared/js.yun/logs/CLAM/TCGA-lung-results/'
split_dir='/shared/js.yun/data/CLAM_data/TCGA-lung-splits/task_2_tumor_subtyping_100/'
csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-lung.csv'
label_dict='{"LUSC":0,"LUAD":1}'

CUDA_VISIBLE_DEVICES=15 python main.py --drop_out \
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
                                    --subtyping \
                                    --opt adam \
                                    --decay_epoch 300 1000 \
                                    --max_epochs 200 \
                                    --label_dict "$label_dict" \
                                    --attn 'js2'