pkill python
git pull

CHECKPOINT_MODE=$1
world_size=$2
h_failure_probability=$3
config=$4
start_iter=$5
for ((i=0; i<$world_size; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "convergence_training.py" $CHECKPOINT_MODE "cuda:$i" $i $world_size $h_failure_probability $config $start_iter>"out$i.txt")&


done