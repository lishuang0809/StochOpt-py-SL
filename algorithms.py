from pickle import FALSE
import numpy as np
import time
import logging

def update_records_and_print(cnt, loss,loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, #acc_records,
                             total_running_time, epoch_running_time, verbose):
    # Compute full gradient and loss
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    lx = np.mean(loss.val(label, data @ x)) + reg * regularizer.val(x)
    # Update gradient norm, loss and time records
    norm_records.append(np.sqrt(g @ g))
    loss_records.append(lx/loss_x0)
    # y_hat = np.sign(data @ x)
    # acc_records.append(np.sum(y_hat == label)/n)
    total_running_time += epoch_running_time
    time_records.append(total_running_time)  
    # Print progress
    if verbose == 1:
        logging.info(
            "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} | loss {:f} |".format(cnt,
                                                                                            epoch_running_time,
                                                                                            norm_records[-1],
                                                                                            loss_records[-1])
                                                                                            )
    return                                                                            
            


def san(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1, dist=None):
    """
    Stochastic Average Newton method for linear model, projection under a norm that depends on hessian of f_i
    :param loss: loss function object, methods: val, prime, dprime
    :param regularizer: regularizer object, methods: val, prime, dprime
    :param data: numpy array, shape (n_samples, n_features)
    :param label: numpy array, shape (n_samples,)
    :param lr: float, learning rate
    :param reg: float, non-negative
    :param dist: sketching distribution
    :param epoch: int, number of data pass
    :param x_0: numpy array of shape (d,), initial point
    :param tol: float, the algo will be stopped if the norm of gradient is less than this threshold
    :param verbose: 0 or 1; 0 means silence, no events be logged;
    :return: trained model params, a list of gradients' norm
    """
    # Denote n = n_samples, d = n_features, we have model coefficients x \in R^d,
    # and we introduce n auxiliary variables {alpha_i} \in R^d, thus we have totally (n+1) variables.
    # We use a big matrix alphas \in \R^{nxd} to store auxiliary variables.
    n, d = data.shape
    alphas = np.zeros((n, d))  # auxiliary variables, it represents one alpha per row
    x = x_0.copy()  # model
    # initial loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # initial gradient
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    # intantiate records
    norm_records, loss_records  = [np.sqrt(g @ g)], [1.0]
    time_records = [0.0]
    cnt = 0  # track the effective data passes
    epoch_running_time, total_running_time = 0.0, 0.0
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution

    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == n:
            alphas = alphas - np.mean(alphas, axis=0, keepdims=True)  # update all alphas
        else:  # i \in [|0, n-1|]
            dot_i = data[i, :] @ x
            # second-order derivation of (i-1)-th loss
            dprime = loss.dprime(label[i], dot_i)
            diff = alphas[i, :] - loss.prime(label[i], dot_i) * data[i, :] - reg * regularizer.prime(x)
            inv = 1. / (1. + reg * regularizer.dprime(x))
            scaled_data = inv * data[i, :]
            cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
            update = lr * (inv * diff - cte * scaled_data)
            alphas[i, :] -= update  # update i-th alpha
            x += update  # update x
        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
    return x, norm_records, loss_records, time_records #loss_records, 


def sana(loss, regularizer, data, label, reg, epoch, x_0, tol=None, verbose=1):
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients x \in R^d, we have totally (n+1) variables.
    # We use a big matrix alphas \in \R^{nxd} to store auxiliary variables.
    n, d = data.shape
    alphas = np.zeros((n, d))  # initialization, it represents one alpha per row
    x = x_0.copy()
    # initial loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # initial gradient
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    # intantiate records
    norm_records, loss_records  = [np.sqrt(g @ g)], [1.0]
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    iis = np.random.randint(0, n, n * epoch)  # uniform sampling
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()

        dot_i = data[i, :] @ x
        prime, dprime = loss.prime(label[i], dot_i), loss.dprime(label[i], dot_i)
        reg_prime, reg_dprime = regularizer.prime(x), regularizer.dprime(x)
        diff = alphas[i, :] - prime * data[i, :] - reg * reg_prime
        inv = 1. / (((n - 1) / n) + reg * reg_dprime)
        scaled_data = inv * data[i, :]
        cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
        update_w = inv * diff - cte * scaled_data
        new_alpha_i = (dprime * (data[i, :] @ update_w) + prime) * data[i, :] + \
                      reg * (reg_dprime * update_w + reg_prime)
        alphas -= (1 / (n - 1)) * (new_alpha_i - alphas[i, :]).reshape(1, -1)
        alphas[i, :] = new_alpha_i
        x += update_w  # update w

        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
    return x, norm_records, loss_records, time_records


def svrg2(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1, dist=None):
    # New version SVRG2
    n, d = data.shape
    alphas = np.zeros((n, d))  # auxiliary variables， it represents one alpha per row
    avg_alphas = np.mean(alphas, axis=0)
    x = x_0.copy()  # model
    # initial loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # initial gradient
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    # intantiate records
    norm_records, loss_records  = [np.sqrt(g @ g)], [1.0]
    time_records = [0.0]

    cnt = 0  # track the effective data passes
    epoch_running_time, total_running_time = 0.0, 0.0
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution

    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == n:
            alphas = loss.prime(label, data @ x).reshape(-1, 1) * data + reg * regularizer.prime(x)  # update all alphas
            avg_alphas = np.mean(alphas, axis=0)
        else:  # i \in [|0, n-1|]
            dot_i = data[i, :] @ x
            prime, dprime = loss.prime(label[i], dot_i), loss.dprime(label[i], dot_i)
            reg_prime, reg_dprime = regularizer.prime(x), regularizer.dprime(x)
            diff = alphas[i] - prime * data[i, :] - reg * reg_prime - avg_alphas
            inv = 1. / (((n - 1) / n) + reg * reg_dprime)
            scaled_data = inv * data[i, :]
            cte = dprime * (scaled_data @ diff) / (1 + dprime * (data[i, :] @ scaled_data))
            update_w = lr * (inv * diff - cte * scaled_data)
            new_alpha_i = alphas[i] + ((n - 1) / n) * update_w
            alphas -= (1. / n) * update_w.reshape(1, -1)
            alphas[i] = new_alpha_i
            x += update_w

        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
    return x, norm_records, loss_records, time_records


def svrg2_old(loss, regularizer, data, label, lr, reg, dist, epoch, x_0, tol=None, verbose=1):
    # Denote n = n_samples, d = n_features, we introduce n auxiliary variables {alpha_i} \in R^d, plus model
    # coefficients, we have totally (n+1) variables. We use a matrix W \in \R^{(n+1)xd} to store these variables.
    # W[0, :] stores the model, and W[i:0] stores the i-th alpha, where i \in [|1, n|]
    n, d = data.shape
    W = np.zeros((d, n + 1))  # initialization
    W[:, 0] = x_0.copy()
    W[:, 1:] = loss.prime(label, data @ W[:, 0]).reshape(1, -1) * data.T + \
               reg * regularizer.prime(W[:, 0]).reshape(-1, 1)
    avg_alpha = np.mean(W[:, 1:], axis=1)
    # initial loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # initial gradient
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    # intantiate records
    norm_records, loss_records  = [np.sqrt(g @ g)], [1.0]
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    # iis = np.random.randint(0, n + 1, n * epoch)  # uniform sampling
    iis = np.random.choice(n + 1, size=n * epoch, p=dist)  # sampling w.r.t customized distribution
    for iter_cnt, i in enumerate(iis):
        start_time = time.time()
        if i == 0:
            W[:, 1:] = loss.prime(label, data @ W[:, 0]).reshape(1, -1) * data.T + \
                       reg * regularizer.prime(W[:, 0]).reshape(-1, 1)
            avg_alpha = np.mean(W[:, 1:], axis=1)
        else:  # i \in [|1, n|]
            dot_i = data[i - 1, :] @ W[:, 0]
            # second-order derivation of (i-1)-th loss
            dprime = loss.dprime(label[i - 1], dot_i)
            diff = loss.prime(label[i - 1], dot_i) * data[i - 1, :] + reg * regularizer.prime(W[:, 0]) - \
                   W[:, i] + avg_alpha
            inv = 1. / reg * regularizer.dprime(W[:, 0])
            scaled_data = inv * data[i - 1, :]
            cte = lr * dprime * (scaled_data @ diff) / (1 + dprime * (data[i - 1, :] @ scaled_data))
            update = lr * inv * diff - cte * scaled_data
            W[:, 0] -= update  # update x
        epoch_running_time += time.time() - start_time

        # records the norm square of gradient after each data pass
        if (iter_cnt + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             W[:, 0], norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            # print(str(cnt)+"-th Data Pass: ", norm_records[-1])
            if tol is not None and norm_records[-1] <= tol:
                return W[:, 0], norm_records, time_records
    return W[:, 0], norm_records, time_records


def sgd(loss, regularizer, data, label, lrs, reg, epoch, x_0, tol=None, beta = 0.0, verbose=1):
    """
    Stochastic Gradient Descent with an array of learning rates.
    """
    n, d = data.shape
    x = x_0.copy()
    x_old = x.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
#    total_start_time = time.time()
    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # gradient of (i-1)-th data point
        grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        # update
        direction = -lrs[i]* grad_i +beta*(x-x_old)
        x_old = x.copy()
        x+=direction
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lrs[i], reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            #    total_running_time += time.time() - total_start_time
#    print("sgd:")
#    print(total_running_time)
    return x, norm_records, loss_records, time_records


def adam(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None,
         beta1 =0.9, beta2 =0.999, eps = 10**(-8.0), verbose = False):
    """Adam method"""
    n, d = data.shape
    x = x_0.copy()
    m = np.zeros(d)
    v = np.zeros(d)
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
#    total_start_time = time.time()
    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # gradient of (i-1)-th data point
        g = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        m = beta1*m +(1-beta1)*g
        v = beta2*v +(1-beta2)*(g*g)
        mhat= m/(1-beta1**(idx+1))
        vhat= v/(1-beta2**(idx+1))
        direction = lr*mhat/(np.sqrt(vhat) +eps)
        # update
        x -= direction
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            #    total_running_time += time.time() - total_start_time
#    print("adam:")
#    print(total_running_time)
    return x, norm_records, loss_records, time_records

def lamb_scheduler(lamb_schedule, start, end, length):
    r""" Returns a array lambts with lambdas that begin in start and finish in end """
    if lamb_schedule is not False:
        if lamb_schedule == "linear":
            lambts = np.linspace(start, end, num=length)
        elif lamb_schedule == "log":
            alpha = 0.1  #early stopping
            lambts = np.linspace(np.exp(start), (1-alpha)*np.exp(start)+alpha*np.exp(end), num=length)
            lambts = np.log(lambts)
        elif lamb_schedule == "loglog":
            pend = np.exp(np.exp(start))
            lambts = np.linspace(pend, np.exp(start), num=length)
            lambts = np.log(np.log(lambts))
        else: # linear 
            lambts = np.linspace(start, end, num=length)
    return lambts



def spsdam(loss, regularizer, data, label, lr, reg, epoch, x_0, s_0, lamb, lamb_schedule, tol=None, eps=0.001, verbose=1, beta=0.0):
    r"""Dampened Stochastic Polyak solver (SPSDam).
    Based on the projection
    w',  s' = argmin_{w\in\R^d} (1-lmbda)||w - w^t||^2
          + (1-lmbda) (s-s^t)^2+ lmbda s^2
                    subject to value + <grad, w - w^t> <= s
       To which the solution is
          step = (value - (1-lmbda) s)_+) / (||grad||^2 + 1 - lmbda)
          w = w -  step *grad,
          s = (1-lmbda)*(s + step)
       Consequently when lmbda -> 1, this method becomes SPS. """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    s = s_0.copy()
#    z_s = s_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))
    # Set a schedule that starts at 0 and ends at 1.
    iis = np.random.randint(0, n, n * epoch + 1)

    lambts = lamb_scheduler(lamb_schedule, 1.0, 0.0, len(iis))

    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):
        i = iis[idx]
        # set the lambda
        if lamb_schedule is not False:
            lamb = lambts[idx] 
        start_time = time.time()        
        # ith data point
        di = data[i, :] 
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], di @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th data point
        lprime = loss.prime(label[i], di @ x)
        gi = lprime * di + reg * regularizer.prime(x)
        # --------computing the stepsize -------
        stepsize = np.maximum(0.0,loss_i - (1.0 - lamb)*s)/ (np.dot(gi, gi) + 1 - lamb)
        # --------updating the slack -------
        s  = (1-lr*lamb)*s  +(1-lamb)*lr*stepsize
        ## Iterative averaging form of momentum
        z += -stepsize*lr*gi 
        x = beta*x +(1.0-beta)*z  # This adds on momentum to x     
        # ----------------------------------------------------------
        epoch_running_time += time.time() - start_time
        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            
    return x, norm_records, loss_records, time_records

def spsL1(loss, regularizer, data, label, lr, reg, epoch, x_0, s_0, lamb, lamb_schedule, delta, tol=None, eps=0.001, verbose=1, beta=0.0):
    r"""The L1 slack SPS method (SPSL1).
    Based on the projection problem
         w',  s' = argmin_{w\in\R^d} ||w - w^t||^2  + delta^-1 (s-s^t)^2+ 2lmbda |s|
                   subject to value + <grad, w - w^t> <= s,  s >= 0.
      To which the solution is
        step1 = (value-s+\delta\lmbda)_+/ (\delta+ ||grad||^2)
        spsstep = value /||grad||^2
        w  = w - min( step1, spsstep) grad.
        s  =   ( s-lmbda delta + delta *step1)_+
      Consequently as delta -> 0 with s_0 =0 this method becomes the SPS method """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    s = s_0.copy()
#    z_s = s_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))
    
    iis = np.random.randint(0, n, n * epoch + 1)
    lambts = lamb_scheduler(lamb_schedule, 0.0, 1.0, len(iis))
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):
        i = iis[idx]
        if lamb_schedule is not False:
            lamb = lambts[idx] 
        start_time = time.time()        
        # ith data point
        di = data[i, :] 
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], di @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th data point
        lprime = loss.prime(label[i], di @ x)
        gi = lprime * di + reg * regularizer.prime(x)
        # --------computing the stepsize -------
        stepdam = np.maximum(0, loss_i-s+delta*lamb)/ (delta+ np.dot(gi, gi))
        spsstep = loss_i /np.dot(gi, gi)
        stepsize = np.minimum( stepdam, spsstep) 

        # --------updating the slack -------
        s  =   np.maximum(0, s-lamb* delta + delta *stepdam)
        
        ## Iterative averaging form of momentum
        z += -stepsize*lr*gi 
        x = beta*x +(1.0-beta)*z  # This adds on momentum to x     
        # ----------------------------------------------------------
        epoch_running_time += time.time() - start_time
        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            
    return x, norm_records, loss_records, time_records

def sps(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, eps=0.001, verbose=1, sps_max=100, beta=0.0):
    """
    Stochastic Polyak Stepsize. Note: We are using an epsilon 
    added to the denominator together with max-capped step for stability.
    """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
#    total_start_time = time.time()
    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], data[i, :] @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th data point
        grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        # update
        sps_step = ((lr*loss_i)/(grad_i @ grad_i +eps))
        stepsize = np.minimum(sps_step,  sps_max)

        z += -stepsize * grad_i 
        x = beta*x +(1-beta)*z
        # direction = -stepsize * grad_i + beta*(x-z) # Heavy ball form of  momentum 
        # z = x.copy()
        # x += direction
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            #    total_running_time += time.time() - total_start_time
#    print("sps:")
#    print(total_running_time)
    return x, norm_records, loss_records, time_records


def sps2(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, eps=0.001, verbose=1, beta=0.0):
    """
    Second order Stochastic Polyak. Introduced an epsilon (eps) in the denominators to avoid overflow
    """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
#    total_start_time = time.time()
    for idx in range(len(iis)):
        i = iis[idx]
        # import pdb; pdb.set_trace()
        start_time = time.time()
        # update in mathematics is  #TODO: DOULBE CHECK THIS !
        # w =  w - (loss_i/|grad_i|^2)grad_i -
        #      0.5*(loss_i^2)/(|grad_i|^4)*(<hess_grad_i, grad_i> )/
        #     ( |grad_i  - hess_grad_i *loss_i/ |grad_i|^2  |^2) *
        #     (grad_i  - hess_grad_i *(loss_i/ |grad_i|^2))
        # ith data point
        di = data[i, :] 
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], di @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th dataloss.prime(label[i], data[i, :] @ x) point
        lprime = loss.prime(label[i], di @ x)
        gi = lprime * di + reg * regularizer.prime(x)
        # Hessian-grad product of (i-1)-th data point  #TODO: DOUBLE CHECK THIS !
        hess_gi = loss.dprime(label[i], di @ x) *(lprime*di@di + reg * regularizer.prime(x)@di )*di 
        hess_gi += reg * regularizer.dprime(x)*gi
        l_div_gnorm = loss_i/ (gi@gi +eps)
        dir_2nd =  gi -hess_gi*l_div_gnorm
        dir_2nd_norm = dir_2nd@dir_2nd
        ## Iterative averaging form of momentum
        z += -lr*l_div_gnorm*gi  #The update is applied to z variable because we use the iterative
        z += -0.5*lr*(l_div_gnorm**2) *(hess_gi@gi/(dir_2nd_norm+eps))*dir_2nd
        x = beta*x +(1.0-beta)*z  # This adds on momentum 

        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            #    total_running_time += time.time() - total_start_time
#    print("sps2:")
#    print(total_running_time)
    return x, norm_records, loss_records, time_records


def sps2slack(loss, regularizer, data, label, lr, reg, epoch, x_0, s_0, lamb, tol=None, eps=0.001, verbose=1, beta=0.0):
    """
    Second order Stochastic Polyak with slack. Introduced an epsilon (eps) in the denominators to avoid overflow
    """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    s = s_0.copy()
#    z_s = s_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))
    
    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
#    total_start_time = time.time()
    for idx in range(len(iis)):
        i = iis[idx]
        # import pdb; pdb.set_trace()
        start_time = time.time()        
        # ith data point
        di = data[i, :] 
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], di @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th data point
        lprime = loss.prime(label[i], di @ x)
        gi = lprime * di + reg * regularizer.prime(x)
        # Hessian-grad product of (i-1)-th data point  
        hess_gi = loss.dprime(label[i], di @ x) *(lprime*di@di + reg * regularizer.prime(x)@di )*di 
        hess_gi += reg * regularizer.dprime(x)*gi
        
        # ---------Added by Shuang, please double check this -------
        temp1 = np.maximum(0,loss_i - (1-lamb)*s)/(1-lamb + gi@gi)
#        x_half = x - temp1*gi 
        s_half = (1-lamb)* (s + temp1)
        q_it_half = loss_i - temp1* (gi@gi) + 0.5*temp1*temp1*(hess_gi@gi)
        g_q_it_half = gi - temp1*hess_gi
        temp2 = np.maximum(0,q_it_half - (1-lamb)*s_half)/(1-lamb + g_q_it_half@g_q_it_half)
        
        ## Iterative averaging form of momentum
        z += -lr*temp1*gi 
        z += -lr*temp2*g_q_it_half
        x = beta*x +(1.0-beta)*z  # This adds on momentum to x                
        s = (1-lamb)*(s_half+temp2) # No momentum added to s
        # ----------------------------------------------------------
        
        epoch_running_time += time.time() - start_time
        
        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records
            #    total_running_time += time.time() - total_start_time 
            #    print("sps2slack:")
            #    print(total_running_time)
    return x, norm_records, loss_records, time_records




def taps(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, tau=0.0, tau_lr=0.01, beta=0.0, verbose=0.0):
    """
    (Moving) targetted stochastic polyak method. When tau_lr=0.0 coresponds to the TAPS method.
    The lmbda controls the rate at which the target is updated. Setting lmbda =0.5 is the default.
    """
    n, d = data.shape
    x = x_0.copy()
    z = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # acc_records =[0.0]
    # Targets for the loss values
    alpha = np.zeros(n)
    alpha_old = np.zeros(n)
    alpha_mean = 0.0
    # Rescale stepsize for iterate averaging momentum
    lr = lr*(1+beta/(1-beta))
    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # loss of (i-1)-th data point
        loss_i = loss.val(label[i], data[i, :] @ x)  + reg * regularizer.val(x)   
        # gradient of (i-1)-th data point
        grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        # update direction coefficient
        coef_direction = (loss_i-alpha[i])/(grad_i @ grad_i +1)
        updatei = lr*coef_direction #+ beta*(alpha[i] - alpha_old[i])
        alpha_old[i] =alpha[i]
        # alpha_oldi =alpha[i]
        alpha[i] += updatei
        z += -lr*coef_direction* grad_i
        x = beta*x +(1-beta)*z
        # direction = -lr*coef_direction* grad_i   + beta*(x-z)
        # z = x.copy()
        # x += direction
        alpha_mean = alpha_mean +(1/n)*(alpha[i]-alpha_old[i])

        if (idx + 1) % n == 0:
            # Update the total target tau. tau_lr =0 coresponds to TAPS
            tau = (1-tau_lr)*tau + tau_lr*(n/(1+n))*alpha_mean
            # Update targets for loss values
            alpha = alpha +lr*(tau-alpha_mean)
            alpha_mean = alpha_mean+lr*(tau-alpha_mean)
            alpha_old = alpha.copy()
            # Monitoring
            epoch_running_time += time.time() - start_time
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records
    return

def sag(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic average gradient algorithm.
    This function is adapted from the code provided by ***
    """
    n, d = data.shape
    x = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x_0), axis=0) + reg * regularizer.val(x_0)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    # Old gradients
    gradient_memory = np.zeros((n, d))
    y = np.zeros(d)
    for i in range(n): # initialize gradient table
        gradient_memory[i] = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
    y = np.sum(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    # updating records because completed one pass over the data.
    update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                    x, norm_records, loss_records, time_records, 
                    total_running_time, epoch_running_time, verbose) 
    cnt += 1


    for idx in range(len(iis)):
        i = iis[idx]

        start_time = time.time()
        # gradient of (i-1)-th loss
        grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
        # update
        # x -= lr * (grad_i - gradient_memory[i] + y)  not exactly same with Algorithm 1 in SAG paper
        y += grad_i - gradient_memory[i]
        x -= lr * y / n
        gradient_memory[i] = grad_i
        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)   
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records


def svrg(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Stochastic variance reduction gradient algorithm.

    reference: Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, Johnson & Zhang

    Note: for all stochastic methods, we measure the performance as a function of the number of effective passes
    through the data, measured as the number of queries to access single gradient (or Hessian) divided by
    the size of dataset. To have a fair comparison with others methods, for SVRG, we pay a careful attention
    to the step where we do a full pass of dataset at the reference point,
    it means that the effective passes should be added one after this step.
    """
    max_effective_pass = epoch // 2
    n, d = data.shape
    x = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x_0), axis=0) + reg * regularizer.val(x_0)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records, time_records, total_running_time = [np.sqrt(g @ g)], [0.0], 0.0
    loss_records = [1.0]
    effective_pass = 0
    for idx in range(max_effective_pass):

        start_time = time.time()
        x_ref = x.copy()
        tot_grad = np.mean(loss.prime(label, data @ x_ref).reshape(-1, 1) * data, axis=0) + \
                   reg * regularizer.prime(x_ref)
        x -= lr * tot_grad
        epoch_running_time = time.time() - start_time

        update_records_and_print(effective_pass, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, loss_records, time_records

        iis = np.random.randint(low=0, high=n, size=n)
        epoch_running_time = 0.0

        for idx in range(len(iis)):  # inner loop
            i = iis[idx]
            start_time = time.time()
            grad_i = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
            grad_i_ref = loss.prime(label[i], data[i, :] @ x_ref) * data[i, :] + reg * regularizer.prime(x_ref)
            d_i = grad_i - grad_i_ref + tot_grad
            x -= lr * d_i
            epoch_running_time += time.time() - start_time

        effective_pass += 1    
        update_records_and_print(effective_pass, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
        # effective_pass += 1
        # g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        # norm_records.append(np.sqrt(g @ g))
        # total_running_time += epoch_running_time
        # time_records.append(total_running_time)
        # if verbose == 1:
        #     logging.info(
        #         "| end of effective pass {:d} | time: {:f}s | norm of gradient {:f} |".format(effective_pass,
        #                                                                                       epoch_running_time,
        #                                                                                       norm_records[-1]))
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records


def snm(loss, data, label, reg, epoch, x_0, tol=None, verbose=1):
    """
    This part implements the method introduced in the paper
    'Stochastic Newton and Cubic Newton Methods with Simple Local Linear-Quadratic Rates, Kovalev et al.',
    according to the Algorithm 3 (\tau = 1) for GLM case.
    Notice that this method supports only L2 regularizer and generally it has O(d^2) complexity.
    The trick used to lead an efficient Algorithm 3 can not be extended to
    other types of regularizer whose hessian are not identity.
    """
    n, d = data.shape
    x = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * (0.5)*(x@x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * x
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    memory_gamma = data @ x
    memory_alpha, memory_beta = loss.prime(label, memory_gamma), loss.dprime(label, memory_gamma)

    g_ = np.mean(memory_alpha.reshape(-1, 1) * data, axis=0)
    h = np.mean((memory_beta * memory_gamma).reshape(-1, 1) * data, axis=0)
    H = np.sqrt(memory_beta).reshape(-1, 1) * data
    B = np.linalg.pinv(reg * np.eye(d) + (H.T @ H) / n)

    iis = np.random.randint(0, n, n * epoch + 1)
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for idx in range(len(iis)):

        # Update
        i = iis[idx]

        start_time = time.time()

        x = B @ (h - g_)
        gamma = data[i, :] @ x
        alpha, beta = loss.prime(label[i], gamma), loss.dprime(label[i], gamma)
        # update g_, h, B
        g_ += (alpha - memory_alpha[i]) * data[i, :] / n
        h += (beta * gamma - memory_beta[i] * memory_gamma[i]) * data[i, :] / n
        Ba = B @ data[i, :]
        B -= (beta - memory_beta[i]) / (n + (beta - memory_beta[i]) * (data[i, :] @ Ba)) * (
                Ba.reshape(-1, 1) @ Ba.reshape(1, -1))
        # update memory
        memory_gamma[i], memory_alpha[i], memory_beta[i] = gamma, alpha, beta

        epoch_running_time += time.time() - start_time

        if (idx + 1) % n == 0:
            cnt += 1
            update_records_and_print(cnt, loss, loss_x0, lambda x: 0.5*x@x, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
            epoch_running_time = 0.0
            if tol is not None and norm_records[-1] <= tol:
                return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records


def vsn(func, data, label, reg, epoch, x_0, tol=None, verbose=1):
    # TODO
    return


#########################
# Deterministic Algorithm
#########################


def gd(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Vanilla Gradient Descent
    """
    n, d = data.shape
    x = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x_0), axis=0) + reg * regularizer.val(x_0)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]
    cnt = 0
    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for cnt in range(1, epoch + 1):

        start_time = time.time()

        grad = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        x -= lr * grad

        epoch_running_time = time.time() - start_time

        # evaluate
        update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records


def newton(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None, verbose=1):
    """
    Vanilla Newton's Method
    """
    n, d = data.shape
    x = x_0.copy()
    # init loss
    loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
    # init grad
    g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
    norm_records = [np.sqrt(g @ g)]
    loss_records = [1.0]

    time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
    for cnt in range(1, epoch + 1):

        start_time = time.time()

        grad = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
        h = np.sqrt(loss.dprime(label, data @ x)).reshape(-1, 1) * data
        hess = reg * np.diag(regularizer.dprime(x)) + (h.T @ h) / n
        # x -= lr * np.linalg.lstsq(hess, grad, rcond=None)[0]
        x -= lr * np.linalg.pinv(hess) @ grad

        epoch_running_time = time.time() - start_time

        # evaluate
        update_records_and_print(cnt, loss, loss_x0, regularizer, data, label, lr, reg, epoch, 
                             x, norm_records, loss_records, time_records, 
                             total_running_time, epoch_running_time, verbose)
        if tol is not None and norm_records[-1] <= tol:
            return x, norm_records, loss_records, time_records

    return x, norm_records, loss_records, time_records
