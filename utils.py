import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import pickle


def plot_iter(result_dict, problem, title, save_path, threshold=1e-8, gradplot=True, fontsize=35):
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    plt.figure(figsize=(9, 8), dpi=1200)

    markers = ["^-", "d-", "*-", ">-", "X-", "o-" , "1-", "2-", "3-", "4-", "8-", "s-"]
    for algo_name, marker in zip(result_dict.keys(), markers):
        result = result_dict[algo_name]
        # result is a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
        len_cut = len(min(result, key=len))
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        # plot
        val_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = np.argmax(val_avg <= threshold) + 1 if np.sum(val_avg <= threshold) > 0 else len(val_avg)
            val_avg =val_avg[:len_cut]
        val_min = np.min(result, axis=0)[:len(val_avg)]
        val_max = np.max(result, axis=0)[:len(val_avg)]
        # grad_std = np.std(result, axis=0)
        # val_min = np.add(val_avg, -grad_std)
        # val_max = np.add(val_avg, grad_std)
        plt.semilogy(np.arange(len(val_avg)), val_avg, marker, label=algo_name, lw=3, markevery=4, markersize=12)
        plt.fill_between(np.arange(len(val_avg)), val_min, val_max, alpha=0.2)

    plt.tick_params(labelsize=35)
    plt.legend(fontsize=fontsize-5,loc='lower right') 
    plt.xlabel("Effective Passes", fontsize=35)
    if gradplot:
        plt.ylabel(r"$\| \nabla f \|_2$", fontsize=35)
    else:
        plt.ylabel(r"$ f(x^k)/f(x^0)$", fontsize=35)    
    plt.title(title, fontsize=35)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + ".pdf"), bbox_inches='tight', pad_inches=0.01)


#def plot_grad_time(result_dict, problem, title, save_path, threshold=1e-8, fontsize=30):
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='sans-serif')
#    plt.figure(figsize=(9, 8), dpi=1200)
    
#    markers = ["^-", "d-", "*-", ">-", "+-", "o-" , "1-", "2-", "3-", "4-", "8-", "s-"]
#    for algo_name, marker in zip(result_dict.keys(), markers):
#        result = result_dict[algo_name]
        # result is a 2-d list with different length,
        # cut it with min_len and convert it to numpy array for plot
#        len_cut = len(min(result, key=len))
#        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        # plot
#        val_avg = np.mean(result, axis=0)
#        if threshold:
#            len_cut = np.argmax(val_avg <= threshold) + 1 if np.sum(val_avg <= threshold) > 0 else len(val_avg)
#            val_avg =val_avg[:len_cut]
#        val_min = np.min(result, axis=0)[:len(val_avg)]
#        val_max = np.max(result, axis=0)[:len(val_avg)]
#        plt.plot(np.arange(len(val_avg)), val_avg, marker, label=algo_name, lw=2)
#        plt.fill_between(np.arange(len(val_avg)), val_min, val_max, alpha=0.2)
        
#    plt.tick_params(labelsize=20)
#    plt.legend(fontsize=fontsize)
#    plt.xlabel("Regularization", fontsize=25)    
#    plt.ylabel("Running Time", fontsize=25)    
#    plt.title(title, fontsize=25)
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    plt.savefig(os.path.join(save_path, title + ".pdf"), bbox_inches='tight', pad_inches=0.01)
            


def run_algorithm(algo_name, algo, algo_kwargs, n_repeat):
    logging.info("------START {}------".format(algo_name))
    grad_iter, loss_iter, grad_time = [], [],  []
    for i in range(n_repeat):
        logging.info("{}-th repetition:".format(i + 1))
        final_w, norm, losses, times = algo(**algo_kwargs)
        grad_iter.append(norm)
        loss_iter.append(losses)
        grad_time.append(times)
    logging.info("------END {}------".format(algo_name))
    # print("MAE: {}".format(np.mean(np.abs(algo_kwargs['data']@final_w - algo_kwargs['label']))))
    return grad_iter, loss_iter, grad_time


def save(path_grad_iter, grad_iter, path_loss_iter, loss_iter, path_grad_time, grad_time):
    with open(path_grad_iter, 'wb') as fp:
        pickle.dump(grad_iter, fp)
    with open(path_grad_time, 'wb') as fp:
        pickle.dump(grad_time, fp)
    with open(path_loss_iter, 'wb') as fp:
        pickle.dump(loss_iter, fp)

def load(path_grad_iter, path_loss_iter, path_grad_time):
    grad_iter, loss_iter, grad_time = None, None, None
    if os.path.isfile(path_grad_iter):
        with open(path_grad_iter, 'rb') as fp:
            grad_iter = pickle.load(fp)
    if os.path.isfile(path_loss_iter):
        with open(path_loss_iter, 'rb') as fp:
            loss_iter = pickle.load(fp)
    if os.path.isfile(path_grad_time):
        with open(path_grad_time, 'rb') as fp:
            grad_time = pickle.load(fp)
    return grad_iter, loss_iter, grad_time


def lipschitz_ridge(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / n + reg


def lipschitz_logistic(X, reg):
    n, d = X.shape
    return np.linalg.norm(X, ord=2) ** 2 / (4. * n) + reg


def max_Li_ridge(X, reg):
    return np.max(np.sum(X ** 2, axis=1)) + reg


def max_Li_logistic(X, reg):
    return 0.25*np.max(np.sum(X ** 2, axis=1)) + reg

def compute_L_max(X, reg, loss_type,regularizor_type):
    if loss_type == "L2" and regularizor_type == "L2":
        L_max =  max_Li_ridge(X, reg)
    elif loss_type == "Logistic" and regularizor_type == "L2":
        L_max =  max_Li_logistic(X, reg)
    else:
        return None
    return L_max
