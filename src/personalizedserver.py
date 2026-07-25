import copy
import torch
import numpy as np
from tqdm import tqdm
from src.personalizedclient import Client
import matplotlib.pyplot as plt
from utils.get_dataset import get_dataset
import time

class Server:
    """Central server managing the HDFL process and differential aggregation."""
    def __init__(self, device, local_model, args, logger=None):
        self.device = device
        self.args = args
        self.get_global_dataset(self.args)
        self.total_clients = self.args.num_users
        self.indexes = [i for i in range(self.total_clients)]
        self.logger = logger

        # Initialize clients with their respective data partitions
        self.clients = [Client(device=device, local_model=copy.deepcopy(local_model), train_dataset=self.train_dataset,
                               test_dataset=self.test_dataset, train_idxs=self.train_user_groups[idx],
                               test_idxs=self.test_user_groups[idx], args=args, index=idx, logger=logger) for idx in
                        self.indexes]

        self.best_accuracy_global_after = 0

    def get_global_dataset(self, args):
        self.train_dataset, self.test_dataset, self.train_user_groups, self.test_user_groups = get_dataset(args)
        self.global_test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.local_bs,
                                                                  shuffle=False)

    def average_weights(self):
        """
        Global Consensus for Backbone:
        Aggregates only the shared backbone parameters (theta) via FedAvg.
        Excludes personalized recalibration parameters (phi).
        """
        print('Aggregating backbone weights...')
        self.logger.info('Aggregating backbone weights...')

        ref_state_dict = self.clients[0].local_model.state_dict()
        w_avg = copy.deepcopy(ref_state_dict)

        for key in w_avg.keys():
            w_avg[key] = torch.zeros_like(w_avg[key])

        for key in w_avg.keys():
            # Skip personalized parameters during global aggregation
            if 'phi' in key:
                continue

            for client in range(self.args.num_users):
                w_avg[key] += self.clients[client].local_model.state_dict()[key]

            w_avg[key] = torch.div(w_avg[key], float(self.args.num_users))

        print('Backbone aggregation complete.')
        self.logger.info('Backbone aggregation complete.')
        return w_avg

    def send_parameters(self, w_avg):
        """
        Broadcasts the aggregated global backbone to all clients,
        while strictly preserving each client's local personalized parameters (phi).
        """
        print('Sending backbone parameters to clients...')
        self.logger.info('Sending backbone parameters to clients...')

        for client in range(self.args.num_users):
            w_local = copy.deepcopy(self.clients[client].local_model.state_dict())

            for key in w_avg.keys():
                # Only overwrite non-personalized parameters
                if 'phi' not in key:
                    w_local[key] = copy.deepcopy(w_avg[key])

            self.clients[client].local_model.load_state_dict(w_local)

        print('Parameter distribution complete.')
        self.logger.info('Parameter distribution complete.')
        return

    def train(self):
        """Main Federated Learning Loop."""
        train_losses = []
        test_losses_global_after = []
        test_acc_global_after = []
        time_history = []
        total_time = 0

        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')
            self.logger.info(f'Start Training round: {epoch}')
            local_train_losses = []
            local_test_losses_global_after = []
            local_test_acc_global_after = []

            idxs = np.random.choice(self.indexes, max(int(self.args.frac * self.total_clients), 1), replace=False)

            # Local Training Phase
            for client in idxs:
                loss, train_time = self.clients[client].train()
                local_train_losses.append(loss)
                total_time += train_time

            local_train_losses_avg = sum(local_train_losses) / len(local_train_losses)
            train_losses.append(local_train_losses_avg)

            # Differential Aggregation Phase
            w_avg = self.average_weights()
            self.send_parameters(w_avg)

            # Evaluation Phase
            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()
                local_test_acc_global_after.append(copy.deepcopy(acc))
                local_test_losses_global_after.append(copy.deepcopy(loss))

            test_losses_global_after.append(sum(local_test_losses_global_after) / len(local_test_losses_global_after))
            test_acc_global_after.append(sum(local_test_acc_global_after) / len(local_test_acc_global_after))
            time_history.append(total_time)

            if test_acc_global_after[-1] >= self.best_accuracy_global_after:
                self.best_accuracy_global_after = test_acc_global_after[-1]
                self.best_epoch = epoch
                self.best_time = total_time

            print(f'Communication Round: {epoch}')
            print(f'Avg training Loss: {train_losses[-1]}')
            print(f'Avg testing Loss. personalized:{test_losses_global_after[-1]}')
            print(f'Avg training Accuracy. personalized after agg:{test_acc_global_after[-1]}')
            print(f'Testing Acc for each client: {local_test_acc_global_after}')
            print(f'Best Accuracy up to now. personalized after agg:{self.best_accuracy_global_after}')
            print(f'Best time: {self.best_time}  Best epoch: {self.best_epoch}')

            self.logger.info(f'Communication Round: {epoch}')
            self.logger.info(f'Avg training Loss: {train_losses[-1]}')
            self.logger.info(f'Avg testing Loss. personalized:{test_losses_global_after[-1]}')
            self.logger.info(f'Avg training Accuracy. personalized after agg:{test_acc_global_after[-1]}')
            self.logger.info(f'Testing Acc for each client: {local_test_acc_global_after}')
            self.logger.info(f'Best Accuracy up to now. personalized after agg:{self.best_accuracy_global_after}')
            self.logger.info(f'Best time: {self.best_time}  Best epoch: {self.best_epoch}')

        self.train_losses = train_losses
        self.test_losses = test_losses_global_after
        self.test_acc = test_acc_global_after
        self.time_history = time_history
        return


