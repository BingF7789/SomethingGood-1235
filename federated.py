import threading
import datetime
import torch
import time
import numpy as np
from BResidual import BResidual
from models import CNNMnist
from optimiser import SGD
from util import sd_matrixing, PiecewiseLinear, trainable_params, StatsLogger


class Cifar10FedEngine:
    def __init__(self, args, dataloader, global_param, server_param, local_param,
                 outputs, cid, tid, mode, server_state, client_states):
        self.args = args
        self.dataloader = dataloader

        self.global_param = global_param
        self.server_param = server_param
        self.local_param = local_param
        self.server_state = server_state
        self.client_state = client_states

        self.client_id = cid
        self.outputs = outputs
        self.thread = tid

        self.mode = mode

        self.model = self.prepare_model()
        # self.threadLock = threading.Lock()

        self.m1, self.m2, self.m3, self.reg1, self.reg2 = None, None, None, None, None

    def prepare_model(self):
        if self.args.dataset == "cifar10":
            model = BResidual(3)
        elif self.args.dataset == "mnist":
            # model = CNNMnist(self.args)
            model = BResidual(1)
        else:
            print("Unknown model type ... ")
            model = None

        model.set_state(self.global_param, self.local_param)
        return model

    def run(self):
        self.model.to(self.args.device)

        if self.args.agg == "scaffold":
            self.server_state["c"] = [t.to(self.args.device) for t in self.server_state["c"]]
            self.client_state["c"] = [t.to(self.args.device) for t in self.client_state["c"]]
            self.client_state["c_i"] = [t.to(self.args.device) for t in self.client_state["c_i"]]
            if self.client_state["c_i_delta"] is not None:
                self.client_state["c_i_delta"] = [t.to(self.args.device) for t in self.client_state["c_i_delta"]]

        output = self.client_run()
        self.free_memory()

        return output

    def client_run(self):
        lr_schedule = PiecewiseLinear([0, 5, self.args.client_epochs], [0, 0.4, 0.001])
        lr = lambda step: lr_schedule(step / len(self.dataloader)) / self.args.batch_size
        opt = SGD(trainable_params(self.model), lr=lr, momentum=0.9, weight_decay=5e-4
                * self.args.batch_size, nesterov=True)

        mean_loss = []
        mean_acc = []
        t1 = time.time()
        c_state = None

        if self.mode == "Train":
            # training process
            for epoch in range(self.args.client_epochs):
                stats = self.batch_run(True, opt.step)
                mean_loss.append(stats.mean('loss'))
                mean_acc.append(stats.mean('correct'))

                # log = "Train - Epoch: " + str(epoch) + ' train loss: ' + str(stats.mean('loss')) +\
                #       ' train acc: ' + str(stats.mean('correct'))
                # self.logger(log, True)

            if self.args.agg == "scaffold":
                c_i_c = tuple(parm_1-parm_2 for parm_1, parm_2 in zip(self.client_state["c_i"], self.client_state["c"]))
                with torch.autograd.no_grad():
                    model_delta = compute_scaffold_delta(self.model.get_state()[0], self.global_param, self.args.device)
                    new_c_i = []
                    for param_1, param_2 in zip(c_i_c, [v for _, v in model_delta.items()]):
                        new_c_i.append(
                            param_1 - param_2 / 0.2 / self.args.client_epochs / len(self.dataloader))
                    new_c_i = tuple(new_c_i)
                    c_i_delta = []
                    for param_1, param_2 in zip(new_c_i, self.client_state["c_i"]):
                        c_i_delta.append(param_1 - param_2)
                    c_i_delta = tuple(c_i_delta)

                c_state = {"global_round": self.client_state["global_round"],
                           "model": None,
                           "model_delta": model_delta,
                           "c_i": new_c_i,
                           "c_i_delta": c_i_delta,
                           "c": None}

        elif self.mode == "Test":
            # validation process
            stats = self.batch_run(False)
            mean_loss.append(stats.mean('loss'))
            mean_acc.append(stats.mean('correct'))

            # log = 'Test - test loss: ' + str(stats.mean('loss')) + ' test acc: ' \
            #       + str(stats.mean('correct'))
            # self.logger(log)

        time_cost = time.time() - t1
        log = self.mode + ' - Thread: {:03d}, Client: {:03d}. Average Loss: {:.4f},' \
                          ' Average Accuracy: {:.4f}, Total Time Cost: {:.4f}'
        self.logger(log.format(self.thread, self.client_id, np.mean(mean_loss), np.mean(mean_acc),
                              time_cost), True)

        self.model.to("cpu")
        output = {"params": self.model.get_state(),
                  "time": time_cost,
                  "loss": np.mean(mean_loss),
                  "acc": np.mean(mean_acc),
                  "client_state": self.client_state,
                  "c_state": c_state}

        # self.outputs[self.thread] = output
        return output

    def batch_run(self, training, optimizer_step=None, stats=None):
        stats = stats or StatsLogger(('loss', 'correct'))
        self.model.train(training)
        for batch in self.dataloader:
            output = self.model(batch)
            # TODO: loss需要检查重新写， sum（） 提到这里
            output['loss'] = self.criterion(output['loss'], self.mode)
            stats.append(output)
            if training:
                output['loss'].sum().backward()
                optimizer_step()
                self.model.zero_grad()
            batch["input"].to("cpu")
            batch["target"].to("cpu")
        return stats

    def criterion(self, loss, mode):
        if self.args.agg == "avg":
            pass

        elif self.args.agg == "prox":
            self.m1 = sd_matrixing(self.model.get_state()[0]).reshape(1, -1).to(self.args.device)
            self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.args.device)
            self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
            loss = loss + 0.3 * self.reg1

        elif self.args.agg == "scaffold":
            c_i_c = tuple(parm_1 - parm_2 for parm_1, parm_2 in zip(self.client_state["c_i"], self.client_state["c"]))
            loss_linear = 0.
            s_dict = [v for _, v in self.model.get_state()[0].items()]
            for param_1, param_2 in zip(s_dict, c_i_c):
                loss_linear -= torch.sum(param_1 * param_2)
            loss += loss_linear

        elif self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
            self.m1 = sd_matrixing(self.model.get_state()[0]).reshape(1, -1).to(self.args.device)
            self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.args.device)
            self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.args.device)
            self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
            self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
            loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2
        return loss

    def free_memory(self):
        if self.m1 is not None:
            self.m1.to("cpu")
        if self.m2 is not None:
            self.m2.to("cpu")
        if self.m3 is not None:
            self.m3.to("cpu")
        if self.reg1 is not None:
            self.reg1.to("cpu")
        if self.reg2 is not None:
            self.reg2.to("cpu")
        if self.args.agg == "scaffold":
            [t.to("cpu") for t in self.server_state["c"]]
            [t.to("cpu") for t in self.client_state["c"]]
            [t.to("cpu") for t in self.client_state["c_i"]]
            if self.client_state["c_i_delta"] is not None:
                [t.to("cpu") for t in self.client_state["c_i_delta"]]

        torch.cuda.empty_cache()

    def logger(self, buf, p=False):
        if p:
            print(buf)
        # self.threadLock.acquire()
        with open(self.args.logDir, 'a+') as f:
            f.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
        # self.threadLock.release()


def compute_scaffold_delta(model_1, model_2, device):
    """
    for scaffold method only
    :param model_1:
    :param model_2:
    :return:
    """

    for key, _ in model_1.items():
        model_1[key] = model_1[key].to(device) - model_2[key].to(device)

    return model_1
