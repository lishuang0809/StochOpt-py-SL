#!/bin/sh

# --type: int, type of problem, 1 means classification and 2 means regression
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
# --run_newton False --run_gd False  --run_alig True
# "colon-cancer" "mushrooms" "duke" "news20.binary" "gisette-scale" "cod-rna.t" "phishing"
python main.py --type 1 --dataset 'phishing' --data_path './datasets/phishing' \
               --name 'phishing' --result_folder 'san' --log_file 'log1.txt' --scale_features False \
               --epochs 50 --n_repetition 5 --reg_power_order 1.0 --tol 0.0000001 \
               --loss "Logistic" --regularizer 'L2'  \
               --run_sgd True --run_san True --run_svrg True \
               --run_adam True --run_sag True --b 1