#!/bin/bash
# Author : Luis Mata
# Date: 2021-12-26
# Description:
#   This script is used to test the Lunar Lander environment and explore the
#   hyperparameters space.

echo 'Activating gym environment...'
set -e
conda activate gym

echo 'Running all experiments...'
time python main_dqn.py -exp_name lunar_lander_dqn_000 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.001 \
--epsilon 1.0 --decay 0.00001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_001 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.005 \
--epsilon 1.0 --decay 0.00001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_002 \
--epochs 5000 --batch_size 64 --gamma 0.80 --learning_rate 0.001 \
--epsilon 1.0 --decay 0.00001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_003 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.001 \
--epsilon 1.0 --decay 0.99 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_004 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.001 \
--epsilon 1.0 --decay 0.0001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_005 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.005 \
--epsilon 1.0 --decay 0.0001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_006 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.01 \
--epsilon 1.0 --decay 0.00001955 --epsilon_min 0.01

time python main_dqn.py -exp_name lunar_lander_dqn_007 \
--epochs 5000 --batch_size 64 --gamma 0.99 --learning_rate 0.01 \
--epsilon 1.0 --decay 0.0001955 --epsilon_min 0.01