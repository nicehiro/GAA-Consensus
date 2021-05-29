import logging
import random
import math

import torch
from torch.autograd import Variable

import adversary as adv
from dataloader import DataDistributor
from options import ARGS
from utils import CUDA, calc_accuracy, writer
from worker import Role, Worker

# compute the number of good workers in this system
ARGS.num_v_workers_sim = ARGS.num_workers_sim - ARGS.num_b_workers_sim
ARGS.cuda = not ARGS.no_cuda and torch.cuda.is_available()
ARGS.alpha = 0.5
ARGS.truncated_bptt_step = 5
data_distributor = DataDistributor(
    ARGS.dataset_path, ARGS.dataset, ARGS.batch_size, ARGS.num_workers_sim + 1
)
data_distributor.distribute()
train_loaders = data_distributor.train_loaders
test_loader = data_distributor.test_loader
attack_method = adv.attack_methods[ARGS.attack_method]().attack


def GAA():
    master = Worker(
        -1,
        None,
        None,
        neighbors_n=ARGS.num_workers_sim,
        train_loader=train_loaders[ARGS.num_workers_sim],
        test_loader=test_loader,
        meta_lr=1e-2,
        policy_lr=1e-2,
        dataset=ARGS.dataset,
        missing_labels=None,
        role=Role.NORMAL,
        period=2e8,
        alpha=0.5,
        extreme_mail=None,
        pretense=1e8,
    )

    workers = []
    for i in range(ARGS.num_workers_sim):
        byzantine = True if i < ARGS.num_b_workers_sim else False
        worker = Worker(
            wid=i,
            atk_fn=attack_method if byzantine else None,
            adv_loss=ARGS.adv_loss if byzantine else None,
            neighbors_n=ARGS.num_workers_sim,
            train_loader=train_loaders[i],
            test_loader=test_loader,
            meta_lr=1e-2,
            policy_lr=1e-2,
            dataset=ARGS.dataset,
            missing_labels=None,
            role=Role.TRADITIONAL_ATTACK if byzantine else Role.NORMAL,
            period=2e8,
            alpha=0.5,
            extreme_mail=None,
            pretense=1e8,
        )
        workers.append(worker)

    alpha_iter_no = 0

    # Q-C settings
    QC = {}
    weights = []
    rewards = {}
    step_size = 0.1
    for worker in workers:
        rewards[worker.wid] = 0
        QC[worker.wid] = 1
        weights.append(1 / ARGS.num_workers_sim)

    for i in range(ARGS.max_iter):
        test_accuracy = calc_accuracy(master.meta_model, test_loader)
        writer.add_scalar("data/test_accuracy", test_accuracy, alpha_iter_no)

        inner_step_count = ARGS.optimizer_steps // ARGS.truncated_bptt_step
        for k in range(inner_step_count):
            master.reset()
            for t in range(ARGS.truncated_bptt_step):
                alpha_iter_no += 1
                Q = []
                for worker in workers:
                    # send current parameters \theta_t to each worker
                    worker.copy_meta_params_from(master.meta_model)
                    # receive submitted gradients Q_t
                    Q.append(worker.submit(alpha_iter_no))
                # update alpha using Q-Consensus
                # loss = master.alpha_update(Q)
                # calc rewards
                server_gradient = master.submit(alpha_iter_no)
                for worker in workers:
                    for i in range(100):
                        index = random.randint(0, server_gradient.shape[0] - 1)
                        reward = math.exp(
                            -abs(Q[worker.wid][index][0] - server_gradient[index][0])
                            * 100
                            * (10 + 0.1 * t + k * ARGS.truncated_bptt_step)
                        )
                        rewards[worker.wid] += reward
                    rewards[worker.wid] *= weights[worker.wid]
                rewards_sum = sum(rewards.values())
                # calc weights
                for worker in workers:
                    rewards[worker.wid] = rewards[worker.wid] / rewards_sum
                    # if k > 0.8 * episodes_num:
                    #     step_size -= step_size / (episodes_num * 0.2)
                    QC[worker.wid] += max(step_size, 0) * (
                        rewards[worker.wid] - QC[worker.wid]
                    )
                    Q_sum = sum(QC.values())
                    for tt in range(ARGS.num_workers_sim):
                        weights[tt] = (QC[tt] / Q_sum) + 0.00001
                    weights_sum = sum(weights)
                    for tt in range(ARGS.num_workers_sim):
                        weights[tt] = (
                            weights[tt]
                            / weights_sum
                            * (1 - 1 / (ARGS.num_workers_sim + 1))
                        )
                # set weights
                master.alpha = CUDA(Variable(torch.tensor(weights)))
                # update GAR \theta using GAR
                master.meta_update(Q)

                writer.add_scalars(
                    "data/alpha",
                    {
                        "weight_{0}".format(i): master.alpha[i].data
                        for i in range(len(workers))
                    },
                    alpha_iter_no,
                )
            # test accuracy
            if alpha_iter_no % 100 == 0:
                test_acc = calc_accuracy(master.meta_model, master.test_loader)
                logging.info("Test Set Accuracy: {0}".format(test_acc))
                logging.info("Alpha: {0}".format(master.alpha.data))
                writer.add_scalar("data/test_accuracy", test_acc, alpha_iter_no)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.autograd.set_detect_anomaly(True)
    GAA()
