import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

def load_movielens(train_path, test_path, base_dir=None):

    if base_dir:
        train_path = os.path.join(base_dir, train_path)
        test_path = os.path.join(base_dir, test_path)

    all_user_set = set()
    all_movie_set = set()

    with open(train_path, 'rt') as data:
        for line in data:
            uid, mid, rat, timestamp = line.strip().split('\t')
            all_user_set.add(uid)
            all_movie_set.add(mid)

    with open(test_path, 'rt') as data:
        for line in data:
            uid, mid, rat, timestamp = line.strip().split('\t')
            all_user_set.add(uid)
            all_movie_set.add(mid)

    num_user = len(all_user_set)
    num_movie = len(all_movie_set)

    inp = np.zeros(shape=(num_user, num_movie))
    mask = np.zeros(shape=(num_user, num_movie))

    with open(train_path, 'rt') as data:
        for line in data:
            uid, mid, rat, timestamp = line.strip().split('\t')
            u_idx = int(uid)-1
            m_idx = int(mid)-1
            inp[u_idx][m_idx] = int(rat)-1
            mask[u_idx][m_idx] = 1
            
    inp_test = np.zeros(shape=(num_user, num_movie))
    mask_test = np.zeros(shape=(num_user, num_movie))
    
    with open(test_path, 'rt') as data:
        for line in data:
            uid, mid, rat, timestamp = line.strip().split('\t')
            u_idx = int(uid)-1
            m_idx = int(mid)-1
            inp_test[u_idx][m_idx] = int(rat)-1
            mask_test[u_idx][m_idx] = 1

    mask = np.repeat(mask,5,axis=1)
    mask_test = np.repeat(mask_test,5,axis=1)

    oe = OneHotEncoder(n_values=5,sparse=False)
    inp = oe.fit_transform(inp) * mask
    inp_test = oe.fit_transform(inp_test) * mask_test
    
    return inp, mask, inp_test, mask_test