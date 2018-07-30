exp='maml-pacs-5way-1shot-TEST'
data_dir='/home/vashisht/data/'
output_dir='/home/vashisht/output/'
dataset='pacs'
num_cls=5
num_inst=1
batch=1
m_batch=20
num_updates=15000
num_inner_updates=5
lr='1e-2'
meta_lr='1e-3'
gpu=1
python maml.py $exp --dataset $dataset --data_dir $data_dir --output_dir $output_dir --num_cls $num_cls --num_inst $num_inst --batch $batch --m_batch $m_batch --num_updates $num_updates --num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr --gpu $gpu 2>&1 | tee $output_dir/$exp/$exp
