#!/bin/sh

# --type: int, type of problem, 0 means classification and 1 means regression
# --dataset: str, name of dataset. hint: can be set "artificial"
# --data_path: str, path to load dataset
# --result_folder: str, name of folder to store the results
# -- epochs: int, epochs to run for one algorithm
# --n_repetitions: int, number of times to repeat for one algorithm
# --ill_conditional: int, 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.
# --loss: str, support "Logistic" for classification; "L2" or "PseudoHuber" for regression. default: "L2"
# --regularizer: str, support "L2" or "PseudoHuber". default: "L2"
# --reg: float, regularization parameter, default-None
# --lr: float, learning rate for Stochastic Averaging Newton, default: 1.0
# --run_xx: do we run xx algorithm
# --run_newton False --run_gd False
# "colon-cancer" "mushrooms" "duke" "news20.binary" "gisette-scale" "cod-rna.t" "phishing"
python main.py --type 1 --dataset 'fourclass' --data_path './datasets/fourclass.txt' \
               --name 'fourclass' --result_folder 'results' --log_file 'log.txt' \
               --epochs 100 --n_repetition 10 --reg_power_order 1.0 --tol 0.00001  \
               --loss "Logistic" --regularizer 'L2' --lamb 100 \
               --run_sgd False --run_spsL1 False --run_spsL1eq True --run_spsL2 False \
               --run_adam False --run_alig False --run_spsL2a False 