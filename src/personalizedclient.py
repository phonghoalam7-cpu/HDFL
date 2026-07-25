import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time

import personalized_scale.utils as psu

np.set_printoptions(edgeitems=30)
torch.set_printoptions(edgeitems=30)

class DatasetSplit(Dataset):
    """Abstract dataset wrapper for local data partitioning."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (L_SupCon) for Stage II.
    Shapes the feature manifold for high cohesion and low coupling.
    """
    def __init__(self, tau_con=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.tau_con = tau_con # Temperature coefficient for contrastive learning
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute dot product and apply temperature scaling (tau_con)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.tau_con)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        loss = - (self.tau_con / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Client:
    """Local client trainer implementing the HDFL two-stage optimization."""
    def __init__(self, device, local_model, train_dataset, test_dataset, train_idxs, test_idxs, args, index, logger=None):
        self.device = device
        self.args = args
        self.index = index
        self.local_model = local_model
        self.logger = logger

        self.trainingLoss = None
        self.testingLoss = None
        self.testingAcc = None

        self.trainloader, self.validloader, self.testloader, self.trainloader_full = self.train_val_test(
            train_dataset, list(train_idxs), test_dataset, list(test_idxs))

        # Core Loss Functions
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device) # L_CE
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device) # L_KD
        self.supcon_loss = SupConLoss(tau_con=0.1).to(self.device) # L_SupCon

    def train_val_test(self, train_dataset, train_idxs, test_dataset, test_idxs):
        """Initializes data loaders for the client."""
        trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = None
        testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                batch_size=int(len(test_idxs) / 10), shuffle=False)
        trainloader_full = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=len(train_idxs),
                                      shuffle=False)
        return trainloader, validloader, testloader, trainloader_full

    def sp_loss(self, features_s, features_t):
        """
        Similarity-Preserving Loss (L_SP) for Stage I.
        Enforces topological isomorphism by matching the Gram matrix (G)
        of the local feature manifold with the global one.
        """
        bsz = features_s.shape[0]
        f_s = features_s.view(bsz, -1)
        f_t = features_t.view(bsz, -1)

        # Compute cosine similarity matrix (Gram matrix)
        G_s = torch.mm(f_s, f_s.t())
        G_t = torch.mm(f_t, f_t.t())

        # Normalize to stabilize numerical calculation
        G_s = F.normalize(G_s, p=2, dim=1)
        G_t = F.normalize(G_t, p=2, dim=1)

        # Frobenius norm distance
        loss = (G_s - G_t).pow(2).mean()
        return loss

    def train(self):
        """
        Executes the Two-Stage Alternating Optimization (HDFL):
        Stage I: Structural Distillation (L_CE + alpha*L_KD + gamma*L_SP)
        Stage II: Feature Recalibration (L_CE + lambda_2*L_Sparsity + lambda_1*L_SupCon)
        """
        self.local_model.to(self.device)
        self.local_model.train()
        epoch_loss = []

        def is_bn_param(k: str) -> bool:
            return (k.startswith('bn1.') or '.bn1.' in k or '.bn2.' in k or '.downsample.1.' in k)

        weights = dict(self.local_model.named_parameters())
        scale_params = []
        backbone_params = []

        # Separate parameters for alternating optimization
        for k in weights.keys():
            if 'phi' in k:
                scale_params.append(weights[k])
            elif is_bn_param(k):
                scale_params.append(weights[k])
                backbone_params.append(weights[k])
            else:
                backbone_params.append(weights[k])

        if self.args.optimizer == 'sgd':
            self.optimizer_scale = torch.optim.SGD(scale_params, lr=self.args.lr, momentum=self.args.momentum)
            self.optimizer_backbone = torch.optim.SGD(backbone_params, lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            self.optimizer_scale = torch.optim.Adam(scale_params, lr=self.args.lr)
            self.optimizer_backbone = torch.optim.Adam(backbone_params, lr=self.args.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} is not implemented.")

        start_time = time.time()

        # Isolate a frozen global teacher model for distillation
        self.local_model.cpu()
        global_model = copy.deepcopy(self.local_model)
        self.local_model.to(self.device)
        global_model.to(self.device)
        global_model.eval()
        for p in global_model.parameters(): p.requires_grad = False

        # --- Stage I Hyperparameters (General Knowledge Extraction) ---
        tau_kd = 2.0  # KD Temperature
        alpha = 0.5   # KD Weight
        gamma = 100.0 # SP Weight (scaled up due to small MSE values)

        # --- Stage II Hyperparameters (Local Knowledge Adaptation) ---
        lambda_1 = 0.85 # SupCon Weight
        lambda_2 = 1e-4 # L1 Sparsity Weight

        # =================================================================
        # Stage I: Structural Distillation (Train Backbone)
        # Objective: Stabilize global topology and prevent forgetting.
        # =================================================================
        psu.mark_only_backbone_as_trainable(self.local_model)

        for iter in range(self.args.local_ep - self.args.local_p_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                logits_s, feats_s = self.local_model(images, return_features=True)

                with torch.no_grad():
                    logits_t, feats_t = global_model(images, return_features=True)

                loss_ce = self.criterion(logits_s, labels.long())

                # L_KD: Logit-level distillation
                loss_kl = self.kl_loss(
                    F.log_softmax(logits_s / tau_kd, dim=1),
                    F.softmax(logits_t / tau_kd, dim=1)
                ) * (tau_kd * tau_kd)

                # L_SP: Feature-level similarity preserving
                loss_sp = self.sp_loss(feats_s, feats_t)

                # Synergistic objective for Stage I
                loss = (1 - alpha) * loss_ce + alpha * loss_kl + gamma * loss_sp

                self.optimizer_backbone.zero_grad()
                loss.backward()
                self.optimizer_backbone.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # =================================================================
        # Stage II: Feature Recalibration (Train phi parameters)
        # Objective: Client-specific feature selection and geometry shaping.
        # =================================================================
        psu.mark_only_scale_as_trainable(self.local_model)

        for iter in range(self.args.local_p_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                logits, features = self.local_model(images, return_features=True)

                loss_ce = self.criterion(logits, labels.long())

                # L_SupCon: Shapes feature manifold (high cohesion, low coupling)
                features_norm = F.normalize(features, dim=1)
                loss_supcon = self.supcon_loss(features_norm, labels)

                # L_Sparsity: Prunes redundant channels (denoising)
                l1_reg = 0.0
                for n, p in self.local_model.named_parameters():
                    if 'phi' in n:
                        l1_reg += torch.norm(p, 1)

                # Synergistic objective for Stage II
                loss = loss_ce + lambda_1 * loss_supcon + lambda_2 * l1_reg

                self.optimizer_scale.zero_grad()
                loss.backward()
                self.optimizer_scale.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.trainingLoss = sum(epoch_loss) / len(epoch_loss)
        end_time = time.time()
        self.local_model.to('cpu')
        del global_model

        return sum(epoch_loss) / len(epoch_loss), end_time - start_time

    def inference(self):
        self.local_model.to(self.device)
        self.local_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        count = 1
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.local_model(images, return_features=False)

                batch_loss = self.criterion(outputs, labels.long())
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
                total += len(labels)
                count += 1

        accuracy = correct / total
        loss = loss / count
        self.testingAcc, self.testingLoss = accuracy, loss
        self.local_model.to('cpu')
        return accuracy, loss