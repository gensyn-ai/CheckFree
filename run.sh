pkill python
git pull
rm log*
rm out*
rm ERROR_LOG*
for ((i=$1; i<$2; i=i+1))
do
    touch "log_stats_proj_2_$i.txt"
    touch "log$i.txt"
    touch "ERROR_LOG_$i.txt"
    touch "out$i.txt"
    (sleep 1; python -u "trainer.py" $i $3 $4 >"out$i.txt") &


done