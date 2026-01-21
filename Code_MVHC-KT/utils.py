import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

def dataShow(data):
    print(data.shape)

    users = data.user_id.unique().tolist()
    print(len(users))

    problem_list = data.problem_id.unique().tolist()
    print(len(problem_list))
    problems_len = len(problem_list)

    skill_list = data.skill_id.unique().tolist()
    print(len(skill_list))
    skills_len = len(skill_list)

    return skills_len, problems_len


def print_first_five_items(data,n=5):
    for i, (key, value) in enumerate(data.items()):
        print(f'{key}: {value}')
        if i >= n:
            break

def problemid_skillid_To_key_value(data):
    result = {}
    count_dict = {}

    for index, row in data.iterrows():
        problem_id = row['problem_id']
        skill_id = row['skill_id']

        if pd.notna(skill_id) and skill_id != '':
            if problem_id in result:
                result[problem_id].add(skill_id)
            else:
                result[problem_id] = {skill_id}

    result = {k: list(v) for k, v in result.items()}
    count_dict = {k: len(v) for k, v in result.items()}

    return result, count_dict

def get_user_reverse_traj(users_trajs_dict):
    """Get each user's reversed trajectory according to her complete trajectory"""
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict

def normalized_adj(adj, is_symmetric=True):
    """Normalize adjacent matrix for GCN"""
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj

def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to tensor"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor

def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_users))

    for userID, sessions in sessions_dict.items():
        for poi in sessions:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H

def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """Drop edge on scipy.sparse.csr_matrix"""
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    # generate edge mask
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix

def get_hyper_deg(incidence_matrix):
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''

    # H  = [num_node, num_edge]
    # DV = [num_node, num_node]
    # DV * H = [num_node, num_edge]

    # HT = [num_edge, num_node]
    # DE = [num_edge, num_edge]
    # DE * HT = [num_edge, num_node]

    # hyper_deg = incidence_matrix.sum(1)
    # inv_hyper_deg = hyper_deg.power(-1)
    # inv_hyper_deg_diag = sp.diags(inv_hyper_deg.toarray()[0])

    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def get_all_users_seqs(users_trajs_dict):
    """Get all users' sequences"""
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs






