test_fun=${1}
# dataset=${1}
variational_strategy=${2}
variational_distribution=${3}
n_train=${4}
n_test=${5}
num_inducing=${6}
num_directions=${7}
num_epochs=${8}
batch_size=${9}
model=${10}


if [ ! -d "./logs" ]
then
    mkdir ./logs
fi

python exp_script.py \
    --test_fun ${test_fun} --dataset ${dataset} --variational_strategy ${variational_strategy}\
    --variational_distribution ${variational_distribution} \
    --n_train ${n_train} --n_test ${n_test}\
    --num_inducing ${num_inducing} --num_directions ${num_directions}\
    --num_epochs ${num_epochs} --batch_size ${batch_size} --model ${model}\
    2>&1 | tee logs/a.out_${model}_${test_fun}_train${n_train}_test${n_test}_m${num_inducing}_p${num_directions}_epochs${num_epochs}_${variational_distribution}_${variational_strategy}