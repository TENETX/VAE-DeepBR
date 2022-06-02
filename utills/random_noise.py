import numpy as np
import random


def random_noise(Y, noise_num=3):
    N, M = Y.shape
    for i in range(N):
        neg_idx = np.where(Y[i] == 0)[0]
        if len(neg_idx) < noise_num:
            Y[i, neg_idx] = 1

        else:
            choose_idx = np.random.choice(neg_idx, noise_num, replace=False)
            Y[i, choose_idx] = 1

    return Y


def random_noise_p_r(Y, noise_rate, noise_num=3):
    N, M = Y.shape
    noise_nums = np.zeros((N, 1))
    noise_num_p = int(N * noise_rate)
    rand_idx_p = np.random.permutation(N)
    choose_idx_p = rand_idx_p[:noise_num_p]
    for p in range(noise_num_p):
        i = choose_idx_p[p]
        neg_idx = np.where(Y[i] == 0)[0]
        if len(neg_idx) < noise_num:
            Y[i, neg_idx] = 1
            noise_nums[i] = len(neg_idx)
        else:
            choose_idx = np.random.choice(neg_idx, noise_num, replace=False)
            Y[i, choose_idx] = 1
            noise_nums[i] = noise_num

    return Y, noise_nums


def missing_label3(Y, missing_rate, missing_num=3):
    N, M = Y.shape
    missing_nums = np.zeros((N, 1))
    missing_num_p = int(N * missing_rate)
    rand_idx_p = np.random.permutation(N)
    choose_idx_p = rand_idx_p[:missing_num_p]
    for p in range(missing_num_p):
        i = choose_idx_p[p]
        neg_idx = np.where(Y[i] == 1)[0]
        if len(neg_idx) < missing_num:
            Y[i, neg_idx] = 0
            missing_nums[i] = len(neg_idx)
        else:
            choose_idx = np.random.choice(neg_idx, missing_num, replace=False)
            Y[i, choose_idx] = 0
            missing_nums[i] = missing_num

    return Y, missing_nums


def missing_label2(Y, missing_rate):
    N, M = Y.shape
    for p in range(N):
        i = Y[p]
        neg_idx = np.where(i == 1)[0]
        missing_num = int(len(neg_idx) * missing_rate)
        if missing_num >= len(neg_idx):
            choose_idx = neg_idx
        else:
            choose_idx = np.random.choice(neg_idx, missing_num, replace=False)
        Y[p][choose_idx] = 0

    return Y, 0


def missing_label(Y_r, missing_rate):
    Y = Y_r.T
    N, M = Y.shape
    count = 0
    maxIteration = 50
    factor = 2
    realpercent = 0
    totalDeleteNum = 0
    totalNum = len(np.where(Y == 1)[0])
    while realpercent < missing_rate:
        if maxIteration == 0:
            factor = 1
            maxIteration = 10
            if count == 1:
                break
            count += 1
        else:
            maxIteration -= 1
        for i in range(N):
            index = np.where(Y[i] == 1)
            if len(index[0]) >= factor:
                deleteNum = int(1 + random.random() * (len(index[0]) - 1))
                totalDeleteNum += deleteNum
                realpercent = 1.0 * totalDeleteNum / totalNum
                if realpercent >= missing_rate:
                    break
                if deleteNum > 0:
                    random.shuffle(index[0])
                    for j in range(deleteNum):
                        Y[i][index[0][j]] = 0
    return Y.T, 0


def missing_label4(Y, missing_num_rate, missing_rate=1):
    N, M = Y.shape
    missing_nums = np.zeros((N, 1))
    missing_num_p = int(N * missing_rate)
    rand_idx_p = np.random.permutation(N)
    choose_idx_p = rand_idx_p[:missing_num_p]
    for p in range(missing_num_p):
        i = choose_idx_p[p]
        neg_idx = np.where(Y[i] == 1)[0]
        missing_num = int(missing_num_rate * len(neg_idx))
        if len(neg_idx) < missing_num:
            Y[i, neg_idx] = 0
            missing_nums[i] = len(neg_idx)
        else:
            choose_idx = np.random.choice(neg_idx, missing_num, replace=False)
            Y[i, choose_idx] = 0
            missing_nums[i] = missing_num

    return Y, 0


if __name__ == '__main__':
    n = 10
    Y = (np.random.uniform(low=0.0, high=1.0, size=(n, n)) > 0.5).astype(int)
    print(Y)
    print()
    ind = np.where(Y == 1)[0]
    totalNum1 = len(ind)
    # Y_p = random_noise(Y, 3)
    Y_p, missing_nums = missing_label(Y, 0.3)
    print(Y_p)
    ind2 = np.where(Y_p == 1)[0]
    totalNum2 = len(ind2)
    print(totalNum2 / totalNum1)
