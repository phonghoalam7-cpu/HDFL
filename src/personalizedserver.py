import copy
import torch
import numpy as np
import torch.nn.functional as F  

from tqdm import tqdm
from src.personalizedclient import Client
import matplotlib.pyplot as plt
from utils.get_dataset import get_dataset
import time

class Server:
    def __init__(self, device, local_model, args, logger=None):
        self.device = device
        self.args = args
        self.get_global_dataset(self.args)
        self.total_clients = self.args.num_users
        self.indexes = [i for i in range(self.total_clients)]
        self.logger = logger

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
        print('Aggregating backbone weights...')
        self.logger.info('Aggregating backbone weights...')

        ref_state_dict = self.clients[0].local_model.state_dict()
        w_avg = copy.deepcopy(ref_state_dict)

        for key in w_avg.keys():
            w_avg[key] = torch.zeros_like(w_avg[key])

        for key in w_avg.keys():

            if 'phi_feat' in key or 'phi_logit' in key:
                continue

            for client in range(self.args.num_users):
                w_avg[key] += self.clients[client].local_model.state_dict()[key]

            w_avg[key] = torch.div(w_avg[key], float(self.args.num_users))

        print('Backbone aggregation complete.')
        self.logger.info('Backbone aggregation complete.')
        return w_avg


    def aggregate_personalized_parameters(self, idxs, tau=0.5, beta_residual=0.9):
        print('Collaborating personalized parameters via Attention...')
        self.logger.info('Collaborating personalized parameters via Attention...')
        
        sampled_clients = [self.clients[i] for i in idxs]
        num_sampled = len(sampled_clients)
        

        if num_sampled <= 1:
            return {idx: {k: copy.deepcopy(v) for k, v in self.clients[idx].local_model.state_dict().items() 
                          if 'phi_feat' in k or 'phi_logit' in k} for idx in idxs}

        phi_vectors = []
        phi_state_dicts = []
        

        for client in sampled_clients:
            state_dict = client.local_model.state_dict()
            phi_dict = {k: v for k, v in state_dict.items() if 'phi_feat' in k or 'phi_logit' in k}
            phi_state_dicts.append(phi_dict)
            
            flattened_phi = torch.cat([v.view(-1) for v in phi_dict.values()])
            phi_vectors.append(flattened_phi)
            
        phi_matrix = torch.stack(phi_vectors)
        

        norm_phi_matrix = F.normalize(phi_matrix, p=2, dim=1)
        similarity_matrix = torch.mm(norm_phi_matrix, norm_phi_matrix.t())
        

        attention_weights = F.softmax(similarity_matrix / tau, dim=1)
        

        updated_phi_updates = {}
        for i, global_idx in enumerate(idxs):
            new_phi_dict = {}
            for key in phi_state_dicts[i].keys():
                agg_tensor = torch.zeros_like(phi_state_dicts[i][key])
                for j in range(num_sampled):
                    alpha_ij = attention_weights[i, j].item()
                    agg_tensor += alpha_ij * phi_state_dicts[j][key]
                
                local_tensor = phi_state_dicts[i][key]

                new_phi_dict[key] = beta_residual * local_tensor + (1.0 - beta_residual) * agg_tensor
                
            updated_phi_updates[global_idx] = new_phi_dict
            
        return updated_phi_updates

    def send_parameters(self, w_avg, personalized_updates=None, active_idxs=None):
        print('Sending dual-track parameters to clients...')
        self.logger.info('Sending dual-track parameters to clients...')

        for client in range(self.args.num_users):
            w_local = copy.deepcopy(self.clients[client].local_model.state_dict())

            for key in w_avg.keys():
                if 'phi_feat' not in key and 'phi_logit' not in key:
                    w_local[key] = copy.deepcopy(w_avg[key])
                    
            if personalized_updates is not None and active_idxs is not None:
                if client in active_idxs:
                    for key in personalized_updates[client].keys():
                        w_local[key] = copy.deepcopy(personalized_updates[client][key])

            self.clients[client].local_model.load_state_dict(w_local)

        print('Parameter distribution complete.')
        self.logger.info('Parameter distribution complete.')
        return

    def train(self):
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

            for client in idxs:
                loss, train_time = self.clients[client].train()
                local_train_losses.append(loss)
                total_time += train_time

            local_train_losses_avg = sum(local_train_losses) / len(local_train_losses)
            train_losses.append(local_train_losses_avg)

            w_avg = self.average_weights()

            personalized_updates = self.aggregate_personalized_parameters(idxs, tau=0.5, beta_residual=0.9)

            self.send_parameters(w_avg, personalized_updates, active_idxs=idxs)

            for client in range(self.args.num_users):
                acc, loss = self.clients[client].inference()

                local_test_acc_global_after.append(copy.deepcopy(acc))
                local_test_losses_global_after.append(copy.deepcopy(loss))

            test_losses_global_after.append(
                sum(local_test_losses_global_after) / len(local_test_losses_global_after))
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
