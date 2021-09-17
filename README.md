# StochOpt-py: Stochastic Optimization in Python

This is a package for benchmarking stochastic optimization methods in Python. For the loss function, logistic loss is provided for binary classification problems and pseudo-huber/L2 loss is supported for regression problems. To benchmark our algorithms, we also provide code for [SAG][sag], [SVRG][svrg], [ADAM][adam],  and [SGD] .


## Package Requirements
---

+ numpy >= 1.13
+ matplotlib >= 2.1
+ scikit-learn >= 0.19


## Usage
---

1. Preparing dataset.

	The datasets we used in our experiments are downloaded from [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). Place the data sets into the ./datasets folder

2. Editing the `run.sh` file.

	You need to edit some arguments to run the codes according to your experimental setting. For example,
	```
	python main.py --type 0 --dataset 'phishing' --data_path './datasets/phishing' \
		       --result_folder 'results' --log_file 'log.txt' \
		       --epochs 50 --n_repetition 10 --ill_conditional 2 --lr 1.0 \
		       --loss "L2" --regularizer 'L2'  \
		       --run_san True --run_sag True --run_svrg True --run_snm True
	```

	Explanation of arguments:

	+  --type: int, type of problem, 0 means classification and 1 means regression
	+  --dataset: str, name of dataset
 	+  --data_path: str, path to load dataset
 	+  --result_folder: str, name of folder to store the experimental results
 	+  --log_file: str, name of log file
 	+  --epochs: int, epochs to run for one algorithm
 	+  --n_repetitions: int, number of times to repeat for one algorithm
 	+  --ill_conditional: int, 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.
 	+  --loss: str, support "Logistic" for classification; "L2" or "PseudoHuber" for regression. default: "L2"
 	+  --regularizer: str, support "L2" or "PseudoHuber". default: "L2"
 	+  --reg: float, regularization parameter, default-None. If you set this argument, then ill_conditional will be ignored automatically.
 	+  --lr: float, learning rate for SAN, default: 1.0
 	+  --run_xx, boolean, set True if you want to run *xx* algorithm


3. Running  `./run.sh` in your terminal.


## Reproducing Experiments in `Stochastic Polyak Stepsize with a Moving Target'

1. Download the data sets colon-cancer, duke, mushrooms and phishing from  [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).  Place the data sets into the ./datasets folder 
2. Run './run_datasets.sh' in your terminal

3. To reproduce the heatmaps, see the code `grid_search_motaps.py' and choose a data set by specifying data_name and data_path. Run in terminal `python grid_search_motaps.py'.

[sag]: https://arxiv.org/abs/1309.2388
[svrg]: https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
[snm]: https://arxiv.org/abs/1912.01597
[adam]: https://arxiv.org/abs/1412.6980
[motaps]: Stochastic Polyak Stepsize with a Moving Target
