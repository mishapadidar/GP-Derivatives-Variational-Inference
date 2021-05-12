dataset="citeseer"
expid="10"
python train.py --fastmode --seed 1212 --epochs 1000\
                --lr 0.0025 --weight_decay 1e-4 --hidden 32\
                --dropout 0.2 --dataset ${dataset}\
                --watch_model True --train_percent 0.036\
                --expid ${expid} --lr_sched "none"\
                2>&1 | tee ../runlogs/a.out_${dataset}_${expid}
     