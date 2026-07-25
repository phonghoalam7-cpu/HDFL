# HDFL
Holistic Decoupling of Generalization and Personalization: A Two-Stage Framework Synergizing Structural Distillation and Feature Recalibration

This repository contains the official PyTorch implementation of HDFL (Holistic Decoupling Federated Learning), a novel two-stage framework designed to resolve the inherent generalization-personalization dilemma in Federated Learning under statistical heterogeneity (Non-IID data).

📌 Overview

Personalized Federated Learning (PFL) aims to customize local models to mitigate performance degradation caused by Non-IID data. While parameter decoupling is a common strategy, conventional methods relying on "physical isolation" (e.g., layer partitioning) often fail to achieve thorough functional separation, leaving general and personalized features entangled.

HDFL addresses this by achieving deep decoupling across model architecture, training chronology, and aggregation strategies:

Stage I (General Knowledge Extraction): Consolidates the global manifold structure via structured distillation (Similarity-Preserving Loss) to prevent catastrophic forgetting.

Stage II (Local Knowledge Adaptation): Refines client-specific features through a lightweight Sparse Recalibration mechanism combined with Supervised Contrastive Learning.

Differentiated Aggregation: Employs standard averaging for the global backbone and a temperature-scaled self-attention mechanism for personalized parameters to facilitate fine-grained knowledge sharing.

🚀 Key Features

Two-Stage Alternating Training: Eliminates gradient conflicts between global consensus and local adaptation.

Feature Recalibration (ChannelScale): A lightweight, plug-and-play module that dynamically filters noise and enhances task-specific semantics via sparse gating.

High Performance on Complex Tasks: Demonstrates significant improvements on challenging datasets (CIFAR-100, Tiny-ImageNet) under extreme Non-IID settings.

Robustness to Heterogeneity: Maintains high stability across varying degrees of data skewness (from highly Non-IID to near-uniform).

🛠️ Project Structure

├── models/

│   └── PersonalizedResnet.py   # ResNet backbone integrated with ChannelScale recalibrator

├── src/

│   ├── personalizedclient.py   # Client-side: Two-stage alternating optimization (Stage I & II)

│   └── personalizedserver.py   # Server-side: Differential aggregation (FedAvg & Self-Attention)

├── utils/

│   ├── utils.py                # Parameter freezing/unfreezing utilities

│   ├── get_dataset.py          # Data partitioning (Dirichlet distribution)

│   └── options.py              # Hyperparameter configurations

├── personalized_scale/

│   └── layers.py               # Implementation of the ChannelScale mechanism

├── main.py                     # Main execution script

└── README.md


📦 Requirements

Python >= 3.8

PyTorch >= 1.9.0

Torchvision

NumPy

Matplotlib


🏃‍♂️ Usage

Basic Training

To run the HDFL framework on CIFAR-10 with extreme Non-IID data ($\alpha=0.1$):

python main.py --dataset cifar --model resnet8 --epochs 400 --local_ep 5 --local_p_ep 2 --noniid 1 --alpha 0.1


Key Hyperparameters

--dataset: Dataset to use (cifar, cifar-100, tinyimagenet).

--model: Backbone architecture (e.g., resnet8, resnet10).

--epochs ($T$): Total number of global communication rounds (Default: 400).

--local_ep ($E$): Total local training epochs per round (Default: 5).

--local_p_ep ($E_2$): Epochs assigned to Stage II Feature Recalibration. The remaining epochs ($E - E_2$) are used for Stage I (Default: 2).

--noniid: Enable Non-IID partitioning (Set to 1).

--alpha ($\alpha_{dir}$): Dirichlet distribution concentration parameter for data heterogeneity (e.g., 0.1, 0.5, 1.0).

Output

During training, the program will:

Log training/testing loss and accuracy.

Save visualization plots (Rounds vs. Accuracy, Time vs. Accuracy) in the ./plots/ directory upon completion.
