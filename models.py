import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import namedtuple


class CNNMnist(nn.Module):
    def __init__(self, args):
        # [10, 20, 320, 50]
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, args.hidden[0], kernel_size=5)
        self.conv2 = nn.Conv2d(args.hidden[0], args.hidden[1], kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(args.hidden[2], args.hidden[3])
        self.fc2 = nn.Linear(args.hidden[3], args.num_classes)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        if args.sm == "full":
            self.server_layers = [self.conv1, self.conv2, self.conv2_drop, self.fc1, self.fc2]
            self.local_layers = []
        elif args.sm == "per":
            self.server_layers = [self.conv1, self.conv2, self.conv2_drop]
            self.local_layers = [self.fc1, self.fc2]
        elif args.sm == "lg":
            self.local_layers = [self.conv1, self.conv2, self.conv2_drop]
            self.server_layers = [self.fc1, self.fc2]
        else:
            print("Things Wrong About args.sm")
            print(args.sm)

    def forward(self, inputs):

        x = F.relu(F.max_pool2d(self.conv1(inputs["input"]), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        loss = self.loss(inputs[x, "target"])

        return {"loss": loss}

    def part_freeze(self):
        for layer in self.server_layers:
            for para in layer.parameters():
                para.requires_grad = False

    def get_state(self, mode="full"):
        '''
        per: shallow shared based layers
        lg: shallow shared personalization layers
        :param mode:
        :return:
        '''
        if mode != "full":
            server_named_params = []
            local_named_params = []
            for layer in self.server_layers:
                server_named_params = server_named_params + [i for i in layer.named_parameters()]
            for layer in self.local_layers:
                local_named_params = local_named_params + [i for i in layer.named_parameters()]

            return server_named_params, local_named_params
        else:
            return [i for i in self.named_parameters()], []

    def set_state(self, w_server, w_local, mode="full"):
        if mode != "full":
            pointer = 0
            for l in range(len(self.server_layers)):
                sd = self.server_layers[l].state_dict()
                sd_len = len(sd)
                for key, param in w_server[pointer:pointer+sd_len]:
                    if key in sd.keys():
                        sd[key] = param.clone().detach()
                    else:
                        print("Server layers mismatch at 'set_state' function.")
                self.server_layers[l].load_state_dict(sd)
                pointer += sd_len

            pointer = 0
            for l in range(len(self.local_layers)):
                sd = self.local_layers[l].state_dict()
                sd_len = len(sd)
                for key, param in w_local[pointer:pointer + sd_len]:
                    if key in sd.keys():
                        sd[key] = param.clone().detach()
                    else:
                        print("Local layers mismatch at 'set_state' function.")
                self.local_layers[l].load_state_dict(sd)
                pointer += sd_len
        else:
            sd = self.state_dict()
            for key, param in w_server:
                if key in sd.keys():
                    sd[key] = param.clone().detach()
                else:
                    print("Server layers mismatch at 'set_state' function.")

            for key, param in w_local:
                if key in sd.keys():
                    sd[key] = param.clone().detach()
                else:
                    print("Local layers mismatch at 'set_state' function.")

            self.load_state_dict(sd)
