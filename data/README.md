# Term project for AI607 (2025 Fall): I Saw, I Saved, I... Shopped? Multi-Behavior Prediction in Online Shopping

## Overview

* This is the dataset for the term project for AI607 (2025 Fall).
* The project is about multi-behavior prediction in online shopping.
* There are 21,716 unique users and 7,977 unique items in total.
* Between a user and an item, there can be four types of interactions (labels):
    * *None*, i.e., no interaction (0)
    * *Viewed*, i.e., clicked (1)
    * Viewed and *saved*, i.e., added to cart (2)
    * Viewed, saved, and *purchased* (3)
* Note that in this dataset, for a user, they must view an item before they can save it, and they must save an item before they can purchase it. Therefore, the label number intuitively indicates the extent of the interaction and forms a hierarchy of interaction types, where a larger label number indicates a strictly more extensive interaction. For example, a user-item pair with label 2 (saved) semantically includes label 1 (viewed) as a sub-interaction. Similarly, a user-item pair with label 3 (purchased) semantically includes labels 1 (viewed) and 2 (saved) as sub-interactions.
* Both users and items are represented by consecutive integers **starting from 1** (i.e., 1-indexed).
* Half of the users (10,858) appear in task 1, and the other half (10,858) appear in task 2. The users for the two tasks are disjoint, i.e., a user is either in task 1 or task 2, but not both.
* For each task, all the items and all the users for that task appear at least once in the training set. The items are shared between the two tasks.
* You are allowed to use the data for one task to help you for the other task.

## Task 1: Interaction-level prediction

### Task description
In this task, you need to predict the interaction type (label) between a user and an item.
That is, given a user-item pair, you need to answer a number in {0, 1, 2, 3} as the predicted label.

### Files
There are three files related to this task:

* `task1_train.tsv`: This is the training set. It has 972,601 lines in total. Each line contains (1) user ID, (2) item ID, and (3) interaction type (label), separated by tab ("\t").
    * The training set `task1_train.tsv` only contains positive labels (1, 2, or 3), i.e., it does not contain any user-item pair with label 0 (no interaction).
* `task1_val_queries.tsv`: This is the queries for the validation set. It has 243,148 lines in total. Each line contains (1) user ID and (2) item ID, separated by tab ("\t").
* `task1_val_answers.tsv`: This is the answers for the validation set. It has 243,148 lines in total. Each line contains (1) user ID, (2) item ID, and (3) interaction type (label), separated by tab ("\t").
    * The answers for the validation set `task1_val_answers.tsv` contains all four types of labels (0, 1, 2, or 3).
* `task1_test_queries.tsv`: This is the queries for the test set. It has 243,154 lines in total. Each line contains (1) user ID and (2) item ID, separated by tab ("\t"). The label is not provided for the test set, and you need to predict the label for each user-item pair in the test set.
* There are user-item pairs that do not appear in any of the training, validation, or test sets, and the label for such user-item pairs is 0 (no interaction).

### What to submit

You need to submit a tsv file `task1_test_answers.tsv` with 243,154 lines in total, where each line contains (1) user ID, (2) item ID, and (3) interaction type (label), separated by tab ("\t"). Each line corresponds to a user-item pair in `task1_test_queries.tsv`.
* The answers for the test set (not given; what you need to predict and submit) also contains all four types of labels (0, 1, 2, or 3).
* **The submitted `task1_test_answers.tsv` should have the SAME format as the answers for the validation set `task1_val_answers.tsv`.**

## Task 2: User-level prediction

### Task description

In this task, for each target user, given all the items that the user has interactions with label 2 (viewed and saved) or 3 (viewed, saved, and purchased), for the remaining items which have label 0 (no interaction) or 1 (only viewed), you need to find the top-k items that the user most likely viewed (i.e., label 1), and rank them by the prediction confidence you have, i.e., put the most confident (i.e., most likely) prediction at first.

### Files

* `task2_train.tsv`: This is the training set. It has 319,905 lines in total. Each line contains (1) user ID, (2) item ID, and (3) interaction type (label), separated by tab ("\t").
    * The training set `task2_train.tsv` only contains labels 2 and 3. It is guaranteed that **all** label-2 and label-3 user-item pairs are included in the training set.
* `task2_val_queries.tsv`: This is the queries for the validation set. It has 3,752 lines in total. Each line contains a single number, which is the user ID.
* `task2_val_answers.tsv`: This is the answers for the validation set. It has 374,905 lines in total. Each line contains (1) user ID, (2) item ID, and (3) interaction type (label), separated by tab ("\t").
    * The answers for the validation set `task2_val_answers.tsv` only contains label 1. For each user in the queries for the validation set `task2_val_queries.tsv`, it is guaranteed that **all** label-1 user-item pairs are included in `task2_val_answers.tsv`. Each user has at least 50 items with which they have label-1 interactions.
* `task2_test_queries.tsv`: This is the queries for the test set. It has 3,752 lines in total. Each line contains a single number, which is the user ID. For each user in the test set, you need to predict the top-k items that the user most likely viewed.
    * The users in the queries for the test set `task2_test_queries.tsv` are disjoint from the users in the queries for the validation set `task2_val_queries.tsv`.
    * For each user in the queries for the test set `task2_test_queries.tsv`, it is guaranteed that they have at least 50 items with which they have label-1 interactions.    

### What to submit

You need to submit a tsv file `task2_test_answers.tsv` with 3,752 lines in total, where each line contains (1) user ID and (2) 50 item IDs, separated by tab ("\t").
Each line corresponds to a user in the test set `task2_test.tsv`.
This means that for each user in the test set `task2_test.tsv`, you need to predict the **top-50** items that the user most likely viewed, and you should **rank the items by the prediction confidence you have**, i.e., put the most confident (i.e., most likely) prediction at first.
* **Note that the submitted `task2_test_answers.tsv` should have a DIFFERENT format as the answers for the validation set `task2_val_answers.tsv`.**
