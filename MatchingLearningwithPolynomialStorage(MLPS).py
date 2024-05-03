import numpy as np
from scipy.optimize import linear_sum_assignment    # importing the necessary libraries
import matplotlib.pyplot as plt

def algorithm_3(M, N, theta_hat, n_ij, n, alpha):
    # Initialize  matrices for optimization
    theta_tilde = np.copy(theta_hat)
    Wki_j = np.zeros((M, N))
    ki_j = [[[] for t in range(Wki_j.shape[1])] for s in range(Wki_j.shape[0])]
    for i in range(M):
        for j in range(N):
            row_ind, col_ind = linear_sum_assignment(-theta_tilde) # the line that matters the most! it does all the magic in O(N^3) ! 
            nijmin = 0
            for d in range(len(row_ind)):
                if d == 0:
                    nijmin = n_ij[row_ind[d], col_ind[d]]
                else:
                    if nijmin > n_ij[row_ind[d], col_ind[d]]:
                        nijmin = n_ij[row_ind[d], col_ind[d]]  # finding the edge with the minimum value, hence picking the no of times a particular arm has been picked
            Wki_j[i, j] = np.sum(theta_tilde[row_ind, col_ind]) + alpha * M * np.sqrt((M + 1) * np.log(n) / nijmin)  # the expression that matters, which needs to be maximized
            ki_j[i][j] = [row_ind, col_ind]

    max_index = np.unravel_index(np.argmax(Wki_j), Wki_j.shape)
    max_index = list(max_index)
    return Wki_j[max_index[0], max_index[1]], ki_j[max_index[0]][max_index[1]]

def algorithm_2(M, N, truepairvalues, total_steps, alpha):
    theta_hat = np.zeros((M, N))
    n_ij = np.zeros((M, N))
    meanReward = np.zeros(total_steps + 1)
    for p in range(0, M):
        for q in range(0, N):
            n = (M - 1) * p + q
            theta_hat[p, q] = np.random.normal(truepairvalues[p, q], 1)
            n_ij[p, q] += 1

    W = []
    k = []
    while True:
        n += 1
        curr_armreward = 0
        new_W, new_k = algorithm_3(M, N, theta_hat, n_ij, n, alpha)
        W.append(new_W)
        k.append(new_k)
        best_k = k[W.index(max(W))]  # picking the best arm, based on the maximization expression
        for i, row_indices in enumerate(best_k[0]):
            col_index = best_k[1][i]
            curr_pairreward = np.random.normal(truepairvalues[row_indices, col_index], 1)
            curr_armreward += curr_pairreward
            n_ij[row_indices, col_index] += 1
            theta_hat[row_indices, col_index] += (1 / n_ij[row_indices, col_index]) * (curr_pairreward - theta_hat[row_indices, col_index])
        if(n==1000):       # condition for non-stationarity, changing the truepairvalues matrix say due to some storm or something
          truepairvalues=np.array([[10,  6,  1],[ 6,  2,  3],[ 0,  4,  6]]) 
        if n > total_steps:
            break
        meanReward[n] = meanReward[n - 1] + (1 / (n + 1e-10)) * (curr_armreward - meanReward[n - 1])
    return meanReward

truepairvalues = np.array([[3, 7, 6], [4, 5, 2], [8, 3, 6]])
M, N, totalSteps = 3, 3, 10000
alphas = [0, 0.1, 0.5, 0.7, 1,10,20]

for alpha in alphas:
    meanReward = algorithm_2(M, N, truepairvalues, totalSteps, alpha)
    plt.plot(np.arange(totalSteps + 1), meanReward, linewidth=2, label=f'Alpha={alpha}')

plt.xscale("log")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('results_alpha.png', dpi=300)
plt.show()
