import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import torch
import random
import torchvision.models as torch_model
from torchvision import datasets, transforms

from utils.options import args_parser
from utils.util import set_logger
from src.personalizedserver import Server
import models.PersonalizedResnet as personalized_model


def exp_parameter(args):
    """Logs the core experimental configurations."""
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number: {args.num_users}')
    print(f'Local Epochs (Total E): {args.local_ep}')
    print(f'Personalized Epochs (Stage II E_2): {args.local_p_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Non-IID Distribution: {args.noniid}')
    print(f'Dirichlet Alpha: {args.alpha}')
    print(f'Random Seed: {args.seed}')


def setup_seed(seed):
    """Ensures deterministic execution for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, logger=None):
    """Main training orchestrator."""
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    accList = []

    # Ensure plot output directory exists
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    for t in range(args.repeat):
        if args.seed == -1:
            args.seed = np.random.randint(0, 10000)
        setup_seed(args.seed)

        logFileName = f'./log/HDFL_dataset{args.dataset}_model{args.model}_frac{args.frac}_round{args.epochs}_epoch{args.local_ep}_pep{args.local_p_ep}_optimizer{args.optimizer}_lr{args.lr}_{args.noniid}_{args.alpha}_seed{args.seed}.log'
        logger = set_logger(logFileName)

        # Initialize network backbone (ResNet architecture)
        if args.dataset == 'cifar':
            args.num_classes = 10
            if args.model == 'resnet8':
                local_model = personalized_model.resnet8(num_labels=args.num_classes).to(device)
            else:
                raise NotImplementedError
        elif args.dataset == 'cifar-100':
            args.num_classes = 100
            if args.model == 'resnet8':
                local_model = personalized_model.resnet8(num_labels=args.num_classes).to(device)
            elif args.model == 'resnet10':
                local_model = personalized_model.resnet10(num_labels=args.num_classes).to(device)
            else:
                raise NotImplementedError
        elif args.dataset == 'tinyimagenet':
            args.num_classes = 200
            if args.model == 'resnet8':
                local_model = personalized_model.resnet8(num_labels=args.num_classes).to(device)
            elif args.model == 'resnet10':
                local_model = personalized_model.resnet10(num_labels=args.num_classes).to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        print(f'Model Structure Initialized: {args.model}')
        logger.info(f'Model Structure Initialized: {args.model}')

        # Initialize HDFL Server
        server = Server(device, local_model, args, logger=logger)

        # Execute federated training loop
        server.train()

        # ---------------------------------------------------------
        # Performance Visualization
        # ---------------------------------------------------------
        print('Training finished. Generating performance plots...')
        rounds = range(1, args.epochs + 1)
        accuracies = server.test_acc
        times = server.time_history

        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16,
            'lines.linewidth': 3
        })

        base_filename = f'HDFL_{args.dataset}_{args.model}_ep{args.epochs}_noniid{args.noniid}_alpha{args.alpha}'

        # Plot 1: Communication Efficiency (Rounds vs. Accuracy)
        plt.figure(figsize=(10, 7))
        plt.plot(rounds, accuracies, color='blue', label='Test Accuracy')
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        plt.title(f'HDFL Convergence: Rounds vs Accuracy ({args.dataset})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'./plots/Rounds_vs_Acc_{base_filename}.png', dpi=300)
        plt.close()

        # Plot 2: Computational Efficiency (Time vs. Accuracy)
        plt.figure(figsize=(10, 7))
        plt.plot(times, accuracies, color='red', label='Test Accuracy')
        plt.xlabel('Training Time (s)')
        plt.ylabel('Test Accuracy')
        plt.title(f'HDFL Efficiency: Time vs Accuracy ({args.dataset})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'./plots/Time_vs_Acc_{base_filename}.png', dpi=300)
        plt.close()

        print(f'Plots saved successfully.')

        print('Best Accuracy:', server.best_accuracy_global_after)
        logger.info(f'Best Accuracy: {server.best_accuracy_global_after}')
        accList.append(max(server.test_acc))
        args.seed = -1

    print(f'Repeated {args.repeat} times. Mean Accuracy: {np.mean(accList):.4f}, Std: {np.std(accList):.4f}')


if __name__ == '__main__':
    args = args_parser()
    args.verbose = 0
    exp_parameter(args)
    train(args)