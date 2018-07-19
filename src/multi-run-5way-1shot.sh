runs=5
exp='maml-omniglot-5way-1shot-run'
data_dir='/home/vashisht/data/'
output_dir='/home/vashisht/output/'
dataset='omniglot'
num_cls=5
num_inst=1
batch=1
m_batch=32
num_updates=15000
num_inner_updates=5
lr='1e-1'
meta_lr='1e-3'
gpu=0

session_name='maml_5way_train'

if [[ $* == *--parallel* ]]; then
	tmux new -s $session_name -d
	max_runs=0
	for (( c=1; c<=$runs; c++ ));	do
	    if [[ $max_runs = 0 ]]; then		
		if [[ $c = 1 ]]; then
		    tmux rename-window run$c
		else
		    tmux new-window -n run$c
		fi
	    fi
            tmux send-keys -t $session_name "python maml.py $exp$c --dataset $dataset --data_dir $data_dir --output_dir $output_dir --num_cls $num_cls --num_inst $num_inst --batch $batch --m_batch $m_batch --num_updates $num_updates --num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr --gpu $gpu 2>&1 | tee $output_dir/$exp$c/$exp$c
            && echo Running next script in 5s && sleep 5 && "
	done
	max_runs=$(($max_runs>$runs?$max_runs:$runs))
	for (( c=1; c<=$max_runs; c++ ));	do
		tmux send-keys -t $session_name:run$c " echo Done" C-m
	done
	tmux a -t $session_name
else
	tmux new -s $session_name -d
	for (( c=1; c<=$runs; c++ ));	do
            tmux send-keys -t $session_name "python maml.py $exp$c --dataset $dataset --data_dir $data_dir --output_dir $output_dir --num_cls $num_cls --num_inst $num_inst --batch $batch --m_batch $m_batch --num_updates $num_updates --num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr --gpu $gpu 2>&1 | tee $output_dir/$exp$c/$exp$c && echo Running next script in 5s && sleep 5 && "
        done
        tmux send-keys -t $session_name " echo Done" C-m
        tmux a -t $session_name
fi

