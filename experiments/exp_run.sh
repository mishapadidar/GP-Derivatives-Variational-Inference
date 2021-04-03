dataset="synthetic-Branin" # synthetic/real - dataset name 
# dataset="real-helens"

# exp setups
n_train=10000
n_test=10000
num_inducing=500
num_directions=2
num_epochs=1000
batch_size=512
lr=0.01
lr_ngd=0.1
num_contour_quad=15
watch_model=True
exp_name="TEST"
seed=0
lr_sched="step_lr"
# compare different methods, comment out the chunk if not comparing with this method
# find runlogs in logs folder

model="DSVGP"
variational_strategy="standard"
variational_distribution="standard" 
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}\
                  ${lr} ${lr_ngd} ${num_contour_quad} ${watch_model} ${exp_name} ${seed} ${lr_sched}

model="DSVGP"
variational_strategy="standard"
variational_distribution="NGD"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}\
                  ${lr} ${lr_ngd} ${num_contour_quad} ${watch_model} ${exp_name} ${seed} ${lr_sched}

model="DSVGP"
variational_strategy="CIQ"
variational_distribution="NGD"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}\
                  ${lr} ${lr_ngd} ${num_contour_quad} ${watch_model} ${exp_name} ${seed} ${lr_sched}

# for traditional SVGP, 
# variational_strategy and variational_distribution don't matter, but need to pass in them.
model="SVGP"
variational_strategy="standard"
variational_distribution="standard"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}\
                  ${lr} ${lr_ngd} ${num_contour_quad} ${watch_model} ${exp_name} ${seed} ${lr_sched}
