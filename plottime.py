import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import pickle

#if __name__ == '__plottime__':
    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--regs', action='store', type=float, dest='regs', default=None)
#    parser.add_argument('--betas', action='store', type=float, dest='betas', default=None)

#    regs = np.array(opt.regs)
#    betas = np.array(opt.betas)
regs=np.array([0.001, 0.003, 0.005, 0.007, 0.009])
betas=np.array([0.0, 0.3, 0.5, 0.7])
NUM_regs=len(regs) 
NUM_betas=len(betas)


markers = ["^-", "d-", "*-", ">-", "+-", "o-" , "1-", "2-", "3-", "4-", "8-", "s-"]
fs = 20

for r in range(NUM_betas):   
    beta = betas[r]
    rtime = np.zeros((NUM_regs, 5))
    marker = markers[r]
    for s in range(NUM_regs):
        reg = regs[s]        
        with open(os.path.join('sps2_slack/colon-cancer', 'dict_time_iter_sum_'+'M'+ str(beta)+'-reg'+ str(reg)), 'rb') as fp:
            grad_iter = pickle.load(fp) 
            rtime[s,:] = list(grad_iter.values()) 
    plt.title("colon-cancer-runtime-M"+str(beta),fontsize=fs)
    plt.xlabel("Regularization",fontsize=fs)
    plt.ylabel("Running time",fontsize=fs)
    fig1 = plt.gcf()
    plt.plot(regs, rtime, marker, lw=2)
    plt.rc('xtick', labelsize=fs) 
    plt.rc('ytick', labelsize=fs) 
    plt.gca().legend(('SGD','SP','SP2','SP2slack','ADAM'))   
    plt.show()
    fig1.savefig(os.path.join('sps2_slack/colon-cancer', "colon-cancer-runtime-M"+str(beta)+".pdf"), bbox_inches='tight', pad_inches=0.01)
    