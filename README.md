# Heterogenous Graph Neural Networks for Recommendation and Multi-Behavior Prediction in Online Shopping

## Overview

This repository contains the term project for **AI607 (2025 Fall)** at KAIST, focusing on multi-behavior prediction in online shopping environments. The project explores how users interact with e-commerce platforms through various behaviors such as viewing, saving, and purchasing items.

In this project we solved two tasks: (1) Multi-behavior prediction for user-item pairs, and (2) Producing a list of the Top-k most probable items for a user to view based on save and purchase history. 

We implemented an apporoach based on representation learning combined with Heterogenous GNNs together with to MLP classifier heads to solve both tasks. For or the second task we accomplished domain transfer by utilizing the dataset for the first task to learn user embeddings, which could then be used for predictions for the second user partition. Our approaches achieved 0.5139 Macro-F1 score on the validation set for the first task, and a 0.195 N-DCG score on the validation set for the second task.
