export CUDA_VISIBLE_DEVICES=1

python3 seg_train.py --name 'Seg_DB_orig' --G 'NVDLMED' --AUG  #--dataroot '%path_where_DB_is' #--test_mode 

