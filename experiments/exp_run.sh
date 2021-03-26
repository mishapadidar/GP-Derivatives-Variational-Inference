# dataset, either test_fun or real dataset
dataset="Branin"

# exp setups
n_train=600
n_test=1000
num_inducing=200
num_directions=2
num_epochs=100
batch_size=200

# compare different methods, comment out the chunk if not comparing with this method
# find runlogs in logs folder

model="DSVGP"
variational_strategy="standard"
variational_distribution="standard"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}

model="DSVGP"
variational_strategy="standard"
variational_distribution="NGD"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}

model="DSVGP"
variational_strategy="CIQ"
variational_distribution="NGD"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}

# for traditional SVGP, 
# variational_strategy and variational_distribution don't matter, but need to pass in them.
model="SVGP"
variational_strategy="standard"
variational_distribution="NGD"
sh ./exp_setup.sh ${dataset} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size} ${model}