import argparse


parser = argparse.ArgumentParser(description="GAA RL parameters.")

############################################################################
# Model setup
# RL settings
parser.add_argument(
    "--truncated_bptt_step",
    type=int,
    default=5,
    metavar="N",
    help="step at which it truncates bptt (default: 5). # T in Algorithm 1",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=10,
    metavar="N",
    help="hidden size of the meta optimizer (default: 10)",
)
# No use now. It corresponds what has been discussed in the original paper
parser.add_argument(
    "--num_layers",
    type=int,
    default=2,
    metavar="N",
    help="number of LSTM layers (default: 2)",
)
# 'sym_log','bias_relu','bias_log','linear','tan'
parser.add_argument("--loss", type=str, default="tan", help="which loss to use")

# distributed learning setting
parser.add_argument(
    "--num_workers_sim",
    type=int,
    default=10,
    metavar="N",
    help="the total number of workers to simulate in the distributed learning env.",
)
parser.add_argument(
    "--num_b_workers_sim",
    type=int,
    default=7,
    metavar="N",
    help="the number of Byzantine workers in this system",
)
parser.add_argument(
    "--rule",
    type=str,
    default="rl",
    help="which rule to suppress Byzantine workers. rl or stat",
)
# mean, krum, bulyan, brute, geomed
parser.add_argument(
    "--method",
    type=str,
    default="classical",
    help="which specific method to be selected in classical rule settings. mean, krum, bulyan, brute, geomed",
)
# noise, one_coor_attack
parser.add_argument(
    "--attack_method",
    type=str,
    default="Random",
    help="decide which attack method to use. Random, Max, Switcher.",
)
parser.add_argument(
    "--atk_role",
    type=int,
    default=1,
    help="decide traditional attack or adversarial attack: normal = 0, "
    "tra_atk = 1, adv_atk = 2, miss_label_atk=3",
)
parser.add_argument(
    "--period", type=int, default=1e8, help="the period of role conversion"
)
parser.add_argument(
    "--adv_loss",
    type=str,
    default="adv_loss1",
    help="Define the loss in adversarial attack. See adv_losses.py",
)
parser.add_argument(
    "--qv_leaked",
    type=int,
    default=2,
    help="levels of visibility of QV set to the attackers:"
    "0: The adversary knows the label composition of the QV set. (deprecated)"
    "1: The adversary knows the distribution where the QV set is"
    "   sampled but does not know the exact QV samples."
    "2: The adversary knows the exact QV samples that GAA is currently using.",
)
parser.add_argument(
    "--missing_label_num",
    type=int,
    default=1,
    help="how many types of label is missing in the validation set",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="the probability of chaning role at a period",
)
parser.add_argument(
    "--pretense",
    type=int,
    default=1e8,
    help="the numeber of iterations that the Byzantine workers pretend to be normal",
)


############################################################################
# Control learning process
parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--val_size", type=int, default=1, help="the size of the quasi-validation set"
)
parser.add_argument(
    "--optimizer_steps",
    type=int,
    default=1000,
    metavar="N",
    help="number of meta optimizer steps (default: 1000). "
    "# K in Algorithm 1, which means episode number currently",
)
parser.add_argument(
    "--max_iter",
    type=int,
    default=1,
    metavar="N",
    help="number of outer iteration (default: 100)",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="learning rate for meta-optimizer"
)

########################################################
# Data and log
parser.add_argument(
    "--home",
    type=int,
    default=2,
    help="home dir: HOMEs = {1: data/, "
    "2: /home/mlsnrs/data/data/pxd},"
    '3: "/home/mlsnrs/data/pxd/"',
)
parser.add_argument(
    "--log_dir", type=str, default="runs/default", help="where to log summaries"
)
parser.add_argument(
    "--p",
    "--print-freq",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 100)",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    help="which data set and corresponding model to use",
)
parser.add_argument(
    "--dataset_path", type=str, default="./data/mnist", help="Dataset storage path."
)
parser.add_argument(
    "--io_workers", type=int, default=8, help="the number of workers for loading data"
)
parser.add_argument(
    "--store_path",
    type=str,
    default="geomed_plus_trace.csv",
    help="path for storing the detection result of geomed+",
)
parser.add_argument(
    "--worker_config_path",
    type=str,
    default="worker.config",
    help="in order to deploy the simulation more efficiently",
)
parser.add_argument(
    "--log_group",
    type=str,
    default="default_group",
    help="To distinguish each group of experiments.",
)

####################################################################################
# cuda setting
parser.add_argument("--which_cuda", type=str, default="0", help="which gpu to use")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)

###############################################
# Other unknown settings
parser.add_argument(
    "--lambda1", type=float, default=0, help="the coeff. of reg for temporal continuity"
)
parser.add_argument(
    "--lambda2", type=float, default=0, help="the coeff. of reg for mag. penalty"
)
parser.add_argument("--reg", type=bool, default=False, help="whether regularize")
parser.add_argument(
    "--extreme", type=bool, default=False, help="whether there is an adaptive attacker"
)
parser.add_argument(
    "--fake_update",
    action="store_true",
    default=False,
    help="whether to use the fake update mechanism",
)

ARGS = parser.parse_args()
