import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import adversary as adv
import baseline
from dataloader import DataLoader
from options import ARGS
from utils import CUDA, calc_accuracy, writer
from worker import Role, Worker

# compute the number of good workers in this system
ARGS.num_v_workers_sim = ARGS.num_workers_sim - ARGS.num_b_workers_sim
ARGS.cuda = not ARGS.no_cuda and torch.cuda.is_available()
ARGS.alpha = 0.5
devices = [torch.device("cuda:1")]
ARGS.truncated_bptt_step = 5
dataloader = DataLoader(ARGS.dataset_path, ARGS.dataset, ARGS.batch_size)
train_loader = dataloader.train_loader
test_loader = dataloader.test_loader
attack_method = adv.attack_methods[ARGS.attack_method]().attack


def GAA():
    master = Worker(
        -1,
        None,
        None,
        neighbors_n=ARGS.num_workers_sim,
        train_loader=train_loader,
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
            train_loader=train_loader,
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

    for i in range(ARGS.max_iter):
        test_accuracy = calc_accuracy(master.meta_model, test_loader)
        writer.add_scalar("data/test_accuracy", test_accuracy, alpha_iter_no)

        inner_step_count = ARGS.optimizer_steps // ARGS.truncated_bptt_step
        for k in range(inner_step_count):
            loss_sum = 0
            prev_loss = CUDA(torch.zeros(1))
            master.reset()
            for t in range(ARGS.truncated_bptt_step):
                alpha_iter_no += 1
                Q = []
                for worker in workers:
                    # send current parameters \theta_t to each worker
                    worker.copy_meta_params_from(master.meta_model)
                    # receive submitted gradients Q_t
                    Q.append(worker.submit(alpha_iter_no))
                # update alpha
                loss = master.alpha_update(Q)
                # update GAR \theta using GAR
                master.meta_update(Q, loss)
                # calc l_{GAA}
                if t > 0:
                    loss_sum += loss - Variable(prev_loss)
                prev_loss = loss.data
                writer.add_scalars(
                    "data/alpha",
                    {
                        "weight_{0}".format(i): master.alpha[i].data
                        for i in range(len(workers))
                    },
                    alpha_iter_no,
                )
            # update policy model
            master.policy_update(loss_sum)
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
