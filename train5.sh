# GPU 5, 13

##################################### 230430 ######################################
exp_code='task_2_tumor_typing_only_major_two_CLAM_100'
data_root_dir='/shared/js.yun/data/CLAM_data/'
feature_folder='TCGA-breast-features'
results_dir='/shared/js.yun/logs/CLAM/TCGA-breast-results/'
split_dir='/shared/js.yun/data/CLAM_data/TCGA-breast-splits-tumor-major-two/task_2_tumor_subtyping_100/'
csv_path='/shared/js.yun/CLAM/dataset_csv/TCGA-breast-tumor-major-two.csv'
label_dict='{"Infiltrating Ductal Carcinoma":0,"Infiltrating Lobular Carcinoma":1}'

CUDA_VISIBLE_DEVICES=13 python main.py --drop_out \
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
                                    --attn 'js'