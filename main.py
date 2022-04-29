import os
import argparse
import logging
import time
import numpy as np
import load_data
from algorithms import san, sag, svrg, snm, vsn, sana, svrg2, gd, newton, sps, taps, sgd, adam, sps2, SP2L2p, spsdam, spsL1, SP2L1p, SP2maxp, SP2max, SP2
import utils
import pickle


# Press the green button in the gutter to run the script.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', dest='name',
                        help="name of setup")
    parser.add_argument('--type', action='store', dest='type', type=int, default=0,
                        help="type of problem, 0 means classification and 1 means regression.")
    parser.add_argument('--dataset', action='store', dest='data_set',
                        help="data set name")
    parser.add_argument('--data_path', action='store', dest='data_path',
                        help='path to load data')
    parser.add_argument('--result_folder', action='store', dest='folder',
                        help="folder path to store experiments results")
    parser.add_argument('--log_file', default='log.txt')
    parser.add_argument('--n_repetition', action='store', type=int, dest='n_rounds', default=10,
                        help="number of repetitions run for algorithm")
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=100)
    parser.add_argument('--reg_power_order', action='store', dest='reg_power_order', type=float, default=1.0,
                        help="can be chosen in 1,2,3. 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.")
    parser.add_argument('--loss', default="L2", help="loss function")
    parser.add_argument('--regularizer', default="L2", help="regularizer type")
    parser.add_argument('--scale_features', action='store', type=float, dest='scale_features', default=True)
    parser.add_argument('--reg', action='store', type=float, dest='reg', default=None)
    parser.add_argument('--lamb', action='store', type=float, dest='lamb', default=None)
    parser.add_argument('--lamb_schedule', action='store', default=False, dest='lamb_schedule',
                        help="name of the lamb scheduling")
    # parser.add_argument('--lamb_schedule', default=False,
    #                     type=lambda x: (str(x).lower() in ['sublinear', 'linear', 'log']))
    parser.add_argument('--delta', action='store', type=float, dest='delta', default=None)
    parser.add_argument("--lr", action='store', type=float, dest='lr', default=1.0)
    parser.add_argument("--beta", action='store', type=float, dest='beta', default=0.0)
    parser.add_argument("--tol", action='store', type=float, dest='tol', default=None)
    parser.add_argument('--run_san', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sana', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sag', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_snm', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_vsn', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_gd', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_newton', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sps2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2L2p', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2L1p', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2maxp', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2max', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_spsdam', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_spsL1', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sgd', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_dai', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_adam', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_taps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_motaps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    ## Parameters for TAPS and MOTAPS
    parser.add_argument("--tau", action='store', type=float, dest='tau', default=None)
    parser.add_argument("--tau_lr", action='store', type=float, dest='tau_lr', default=None)   
    parser.add_argument("--motaps_lr", action='store', type=float, dest='motaps_lr', default=None)
    opt = parser.parse_args()
    return opt


def build_problem(opt):
    folder_path = os.path.join(opt.folder, opt.data_set)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.basicConfig(filename=os.path.join(folder_path, opt.log_file),
                        level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    logging.info(opt)

    # load data
    X, y = load_data.get_data(opt.data_path)
    X = X.toarray()  # convert from scipy sparse matrix to dense
    logging.info("Data Sparsity: {}".format(load_data.sparsity(X)))
    if opt.type == 1:
        problem_type = "classification"
    elif opt.type == 2:
        problem_type = "regression"
    else:
        problem_type = "unknown"

    criterion, penalty, X, y = load_data.load_problem(X, y, problem_type,
                loss_type = opt.loss, regularizer_type = opt.regularizer, 
                bias_term = True, scale_features = opt.scale_features)

    if opt.reg is None:
        n, d = X.shape
        # sig_min = np.min(np.diag(X@X.T))
        # reg = sig_min  / (n**(opt.reg_power_order))
        reg = 1.0  / (n**(opt.reg_power_order))
    else:
        reg = opt.reg
    logging.info("Regularization param: {}".format(reg))

    return folder_path, criterion, penalty, reg, X, y 

def run(opt, folder_path, criterion, penalty, reg, X, y):
    # data_set = opt.data_set
    # folder_path = os.path.join(opt.folder, data_set)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # logging.basicConfig(filename=os.path.join(folder_path, opt.log_file),
    #                     level=logging.INFO, format='%(message)s')
    # logging.info(time.ctime(time.time()))
    # logging.info(opt)

      # do we use uniform sampling for SAN?

    # # load data
    # X, y = load_data.get_data(opt.data_path)
    # X = X.toarray()  # convert from scipy sparse matrix to dense
    # logging.info("Data Sparsity: {}".format(load_data.sparsity(X)))
    # if opt.type == 1:
    #     problem_type = "classification"
    # elif opt.type == 2:
    #     problem_type = "regression"
    # else:
    #     problem_type = "unknown"

    # criterion, penalty, X, y = load_data.load_problem(X, y, problem_type,
    #             loss_type = opt.loss, regularizer_type = opt.regularizer, 
    #             bias_term = True, scale_features = opt.scale_features)

    n, d = X.shape
    logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))
    # logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))



    epochs = opt.epochs
    n_rounds = opt.n_rounds
    x_0 = np.zeros(d)  # np.random.randn(d)
    s_0 = np.zeros(1)

    dict_grad_iter = {}
    dict_loss_iter = {}
    dict_time_iter = {}
    if opt.run_svrg2:
        np.random.seed(0)  # random seed to reproduce the experiments
        svrg2_lr = opt.lr  # 0.001*reg
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg2_lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="SVRG2", algo=svrg2, algo_kwargs=kwargs, n_repeat=n_rounds)
 
        # dict_acc_iter["SVRG2"] = grad_iter
        dict_grad_iter["SVRG2"] = grad_iter
        dict_loss_iter["SVRG2"] = loss_iter
        dict_time_iter["SVRG2"] = grad_time
        utils.save(os.path.join(folder_path, 'svrg2_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'svrg2_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'svrg2_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'svrg2_grad_iter'),
                                          os.path.join(folder_path, 'svrg2_loss_iter'),
                                          os.path.join(folder_path, 'svrg2_grad_time'))
        if grad_iter:
            dict_grad_iter["SVRG2"] = grad_iter

    if opt.run_san:
        np.random.seed(0)  # random seed to reproduce the experiments
        is_uniform = True # #TODO: remove this option 
        if is_uniform: # This could be in the initialization of san instead of here?
            dist = None
        else:
            p_0 = 1. / (n + 1)
            logging.info("Probability p_0: {:}".format(p_0))
            dist = np.array([p_0] + [(1 - p_0) / n] * n)

        kwargs = {"loss": criterion, "data": X, "label": y, "lr": opt.lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="SAN", algo=san, algo_kwargs=kwargs, n_repeat=n_rounds)
            # loss_iter,
        dict_grad_iter["SAN"] = grad_iter
        dict_loss_iter["SAN"] = loss_iter
        dict_time_iter["SAN"] = grad_time
        utils.save(os.path.join(folder_path, 'san_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'san_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'san_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'san_grad_iter'),
                                          os.path.join(folder_path, 'san_loss_iter'),
                                          os.path.join(folder_path, 'san_grad_time'))
        if grad_iter:
            dict_grad_iter["SAN"] = grad_iter

    if opt.run_sana:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(algo_name="SANA", algo=sana,
                                                   algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["SANA"] = grad_iter 
        dict_loss_iter["SANA"] = loss_iter
        dict_time_iter["SANA"] = grad_time
        utils.save(os.path.join(folder_path, 'sana_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sana_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'sana_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'sana_grad_iter'),
                                          os.path.join(folder_path, 'sana_loss_iter'),
                                          os.path.join(folder_path, 'sana_grad_time'))
        if grad_iter:
            dict_grad_iter["SANA"] = grad_iter

    if opt.run_vsn:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="VSN", algo=vsn, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["VSN"] = grad_iter
        dict_loss_iter["VSN"] = loss_iter
        dict_time_iter["VSN"] = grad_time
        utils.save(os.path.join(folder_path, 'vsn_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'vsn_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'vsn_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'vsn_grad_iter'),
                                          os.path.join(folder_path, 'vsn_loss_iter'),
                                          os.path.join(folder_path, 'vsn_grad_time'))
        if grad_iter:
            dict_grad_iter["VSN"] = grad_iter

    
    if opt.run_snm:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="SNM", algo=snm, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["SNM"] = grad_iter
        dict_loss_iter["SNM"] = loss_iter
        dict_time_iter["SNM"] = grad_time
        utils.save(os.path.join(folder_path, 'snm_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'snm_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'snm_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'snm_grad_iter'),
                                          os.path.join(folder_path, 'snm_loss_iter'),
                                          os.path.join(folder_path, 'snm_grad_time'))
        if grad_iter:
            dict_grad_iter["SNM"] = grad_iter

    if opt.run_gd:
        # 1/L, L is the smoothness constant
        if opt.loss == "L2" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_ridge(X, reg)
        elif opt.loss == "Logistic" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_logistic(X, reg)
        else:
            print("Warning!!! GD learning rate")
            gd_lr = 0.01
        logging.info("Learning rate used for Gradient descent: {:f}".format(gd_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": gd_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="GD", algo=gd, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["GD"] = grad_iter
        dict_loss_iter["GD"] = loss_iter
        dict_time_iter["GD"] = grad_time
        utils.save(os.path.join(folder_path, 'gd_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'gd_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'gd_grad_time'), grad_time)

    if opt.run_newton:
        newton_lr = 1.0
        logging.info("Learning rate used for Newton method: {:f}".format(newton_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": newton_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="Newton", algo=newton, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["Newton"] = grad_iter
        dict_loss_iter["Newton"] = loss_iter
        dict_time_iter["Newton"] = grad_time
        utils.save(os.path.join(folder_path, 'newton_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'newton_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'newton_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'newton_grad_iter'),
                                          os.path.join(folder_path, 'newton_loss_iter'),
                                          os.path.join(folder_path, 'newton_grad_time'))
        if grad_iter:
            dict_grad_iter["Newton"] = grad_iter

    if opt.run_sgd:
        np.random.seed(0)
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SGD learning rate")
            lr_max = 0.01
        else:
            lr_max = 1.0/(1.0*L_max)

          #lrs = 0.5*lr_max*np.ones(n_iters)

        lrs = lr_max*(1./np.sqrt(np.arange(1, n * epochs + 1)))
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SGD" 
        else:
            beta = opt.beta
            algo_name = "SGDM" + str(beta) 
        logging.info("Learning rate max used for SGD method: {:f}".format(lr_max))
        kwargs = {"loss": criterion, "data": X, "label": y, "lrs": lrs, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "beta": beta}

        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=sgd, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter   
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'sgd_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sgd_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'sgd_grad_time'), grad_time)

    if opt.run_sps:
        np.random.seed(0)
        ## Clever L_max used in MOTAPS paper
        # L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        # sps_max = 4.0/L_max
        sps_max = 100.0 # Nico suggested 10 or 100 or larger. 
        sps_lr = 1.0
        eps = 0.01
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP"
        else:
            beta = opt.beta
            algo_name = "SPM" + str(beta)
        logging.info("Learning rate used for SP method: {:f}".format(sps_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "eps": eps, "sps_max": sps_max, "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=sps, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'sp_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sp_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'sp_grad_time'), grad_time)
    if opt.run_sps2:
        np.random.seed(0)
        sps2_lr =1.0
        eps=0.01
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "eps": eps,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=sps2, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'sp2_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sp2_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'sp2_grad_time'), grad_time)
    
    if opt.run_SP2L2p:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2L2$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2L2$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=SP2L2p, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'SP2L2p_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'SP2L2p_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'SP2L2p_grad_time'), grad_time)


    if opt.run_SP2L1p:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2L1$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2L1$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=SP2L1p, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'SP2L1p_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'SP2L1p_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'SP2L1p_grad_time'), grad_time)


    if opt.run_SP2maxp:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2max$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2max$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=SP2maxp, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'SP2maxp_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'SP2maxp_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'SP2maxp_grad_time'), grad_time)


    if opt.run_SP2max:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2max"
        else:
            beta = opt.beta
            algo_name = "SP2maxM" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=SP2max, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'SP2max_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'SP2max_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'SP2max_grad_time'), grad_time)


    if opt.run_SP2:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2"
        else:
            beta = opt.beta
            algo_name = "SP2M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=SP2, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        dict_time_iter[algo_name] = grad_time
        utils.save(os.path.join(folder_path, 'SP2_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'SP2_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'SP2_grad_time'), grad_time)



    if opt.run_spsdam:
        np.random.seed(0)
        spsdam_lr =1.0
        if opt.lamb is None:
            lamb = 0.98
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SPSdam"
        else:
            beta = opt.beta
            algo_name = "SPSdamM" + str(beta)
        if opt.lamb_schedule is not False:
            algo_name = algo_name + opt.lamb_schedule
        else: 
            algo_name = algo_name + "-l-" + str(lamb)

        kwargs = {"loss": criterion, "data": X, "label": y, "lr": spsdam_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol,  "lamb": lamb, "lamb_schedule": opt.lamb_schedule, "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=spsdam, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        utils.save(os.path.join(folder_path, 'spsdam_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'spsdam_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'spsdam_grad_time'), grad_time)

    if opt.run_spsL1:
        np.random.seed(0)
        spsL1_lr =1.0
        if opt.delta is None:
            delta = 1.0
        else:
            delta = opt.delta
        if opt.lamb is None:
            lamb = 1.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SPSL1"
        else:
            beta = opt.beta
            algo_name = "SPSL1M" + str(beta)
        if opt.lamb_schedule is not False:
            algo_name = algo_name + opt.lamb_schedule
        else: 
            algo_name = algo_name + "-l-" + str(lamb)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": spsL1_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "lamb": lamb, "lamb_schedule": opt.lamb_schedule, "delta": delta, "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=spsL1, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        utils.save(os.path.join(folder_path, 'spsL1_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'spsL1_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'spsL1_grad_time'), grad_time)


    if opt.run_taps:
        np.random.seed(0)
        taps_lr = 1.0
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "TAPS"
        else:
            beta = opt.beta
            algo_name = "TAPSM" + str(beta)
        if opt.tau == None:
            # Getting the optimal function value tau
            from scipy.optimize import fmin_l_bfgs_b
            loss_lambda = lambda x: np.mean(criterion.val(y, X @ x), axis=0) + reg * penalty.val(x)
            grad_lambda = lambda x: np.mean(criterion.prime(y, X @ x).reshape(-1, 1) * X, axis=0) + reg * penalty.prime(x)
            x_init = np.zeros(d)
            x_min, tau, _ = fmin_l_bfgs_b(loss_lambda, x_init, grad_lambda, pgtol=1e-30, factr=1e-30) 
        else: 
            tau = opt.tau
        # print("tau = ", tau)
        logging.info("tau = {:f}".format(tau))
        logging.info("Learning rate used for TAPS method: {:f}".format(taps_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": taps_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty,
                   "tol": opt.tol, "tau": tau, "tau_lr": 0.0, "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=taps, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        utils.save(os.path.join(folder_path, 'taps_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'taps_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'taps_grad_time'), grad_time)

    if opt.run_motaps:
        np.random.seed(0)
        # thum_tau_lr = 0.75*np.exp(-(d/n +1/reg)) + 1e-6
        # thum_tau_lr = 0.9*np.exp(-(d/n +1/reg)) + 1e-6
        # thum_tau_lr = 0.0001+0.1*np.exp(-0.005*d/n -0.002/(reg+0.001))
        gamma_theo = 1.0/(1+0.25*reg*np.exp(reg))
        if(opt.tau_lr == None):
            tau_lr = 0.9 -0.899*gamma_theo #1.0 - gamma_theo
        else:
            tau_lr = opt.tau_lr
        if(opt.motaps_lr == None):
            motaps_lr = gamma_theo
        else:   
            motaps_lr = opt.motaps_lr
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "MOTAPS"
        else:
            beta = opt.beta
            algo_name = "MOTAPSM" + str(beta)
        if opt.tau == None:
            tau = 0.0 
        else: 
            tau = opt.tau
        logging.info("Learning rate used for MOTAPS method: {:f}".format(taps_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": motaps_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty,
                   "tol": opt.tol, "tau": tau, "tau_lr": tau_lr, "beta": beta}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name=algo_name, algo=taps, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter[algo_name] = grad_iter
        dict_loss_iter[algo_name] = loss_iter
        utils.save(os.path.join(folder_path, 'motaps_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'motaps_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'motaps_grad_time'), grad_time)


    if opt.run_adam:
        np.random.seed(0)
        lr = 0.001  # default should be lr = 0.001, but works well with 0.005

        logging.info("Learning rate max used for ADAM method: {:f}".format(lr))
        # loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None,
        #  beta1 =0.9, beta2 =0.999, eps = 10**(-8.0), verbose = False
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol}

        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="ADAM", algo=adam, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["ADAM"] = grad_iter
        dict_loss_iter["ADAM"] = loss_iter
        dict_time_iter["ADAM"] = grad_time
        utils.save(os.path.join(folder_path, 'adam_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'adam_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'adam_grad_time'), grad_time)

    if opt.run_sag:
        np.random.seed(0)  # random seed to reproduce the experiments
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SVRG learning rate")
            sag_lr = 0.01
        else:
            sag_lr = 1.0/(2.0*L_max)
        # in the SAG paper, the lr given by theory is 1/16L.
        # sag_lr = 0.25 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SAG: {:f}".format(sag_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sag_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="SAG", algo=sag, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["SAG"] = grad_iter
        dict_loss_iter["SAG"] = loss_iter   
        utils.save(os.path.join(folder_path, 'sag_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'sag_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'sag_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'sag_grad_iter'),
                                          os.path.join(folder_path, 'sag_loss_iter'),
                                          os.path.join(folder_path, 'sag_grad_time'))
        if grad_iter:
            dict_grad_iter["SAG"] = grad_iter

    if opt.run_svrg:
        np.random.seed(0)  # random seed to reproduce the experiments
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SVRG learning rate")
            svrg_lr = 0.01
        else:
            svrg_lr = 1/(2.0*L_max)

        # in the book "Convex Optimization: Algorithms and Complexity, Sébastien Bubeck",
        # the Theorem 6.5 indicates that the theory choice lr of SVRG should be 1/10L.
        # svrg_lr = 0.4 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SVRG: {:f}".format(svrg_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="SVRG", algo=svrg, algo_kwargs=kwargs, n_repeat=n_rounds)
        dict_grad_iter["SVRG"] = grad_iter
        dict_loss_iter["SVRG"] = loss_iter
        utils.save(os.path.join(folder_path, 'svrg_grad_iter'), grad_iter,
                   os.path.join(folder_path, 'svrg_loss_iter'), loss_iter,
                   os.path.join(folder_path, 'svrg_grad_time'), grad_time)
    else:
        grad_iter, loss_iter, grad_time = utils.load(os.path.join(folder_path, 'svrg_grad_iter'),
                                          os.path.join(folder_path, 'svrg_loss_iter'),
                                          os.path.join(folder_path, 'svrg_grad_time'))
        if grad_iter:
            dict_grad_iter["SVRG"] = grad_iter

## Final return of run()     
    return dict_grad_iter, dict_loss_iter, dict_time_iter #, opt.data_set, opt.name, folder_path

if __name__ == '__main__': 

    opt = get_args()   #get options and parameters from parser
    folder_path, criterion, penalty, reg, X, y  = build_problem(opt)  #build the optimization problem
    dict_grad_iter, dict_loss_iter, dict_time_iter  = run(opt, folder_path, criterion, penalty, reg, X, y)

    #Plot the training loss and gradient convergence
    utils.plot_iter(result_dict=dict_grad_iter, problem=opt.data_set, title = opt.name + "-grad-iter" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path)
    utils.plot_iter(result_dict=dict_loss_iter, problem=opt.data_set, title = opt.name + "-loss-iter" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, gradplot=False)

    # Some code Shuang wrote
    dict_time_iter_sum = {} 
    for key in dict_time_iter: 
        dict_time_iter_sum[key] = sum(sum(np.array(dict_time_iter[key]))) 
    # Some code Shuang wrote
    with open(os.path.join(folder_path, 'dict_time_iter_sum_'+'M'+ str(opt.beta)+'-reg'+ "{:.2e}".format(reg)), 'wb') as fp:
         pickle.dump(dict_time_iter_sum, fp)
      