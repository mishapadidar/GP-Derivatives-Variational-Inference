test_fun="Branin"
# dataset="3droads"

n_train=600
n_test=1000
num_inducing=20
num_directions=2
num_epochs=1000
batch_size=200

# compare different methods
# find the runlogs in logs folder
variational_strategy="standard"
variational_distribution="standard"
sh ./exp_setup.sh ${test_fun} ${variational_strategy} ${variational_distribution}\
                  ${n_train} ${n_test} ${num_inducing}\
                  ${num_directions} ${num_epochs} ${batch_size}

# variational_strategy="standard"
# variational_distribution="NGD"
# sh ./exp_setup.sh ${test_fun} ${variational_strategy} ${variational_distribution}\
#                   ${n_train} ${n_test} ${num_inducing}\
#                   ${num_directions} ${num_epochs} ${batch_size}

# variational_strategy="CIQ"
# variational_distribution="NGD"
# sh ./exp_setup.sh ${test_fun} ${variational_strategy} ${variational_distribution}\
#                   ${n_train} ${n_test} ${num_inducing}\
#                   ${num_directions} ${num_epochs} ${batch_size}