import copy
import torch
import numpy as np
from scipy.linalg import solve


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def outlier_detect(w_global, w_local, itera):
    w_global = w_global['resnet.fc1.weight'].cpu().numpy()
    w = []
    for i in range(len(w_local)):
        temp = (w_local[i]['resnet.fc1.weight'].cpu().numpy() - w_global) * 100
        w.append(temp)
    res = search_neuron_new(w)
    return res

def search_neuron_new(w):
    w = np.array(w)
    pos_res = np.zeros((len(w), 26, 512))
    for i in range(w.shape[1]):
        for j in range(w.shape[2]):
            temp = []
            for p in range(len(w)):
                temp.append(w[p, i, j])
            max_index = temp.index(max(temp))
            # pos_res[max_index, i, j] = 1
            outlier = np.where(np.abs(temp) / abs(w[max_index, i, j]) > 0.8)
            if len(outlier[0]) < 2:
                pos_res[max_index, i, j] = 1
    return pos_res

def whole_determination(pos, w_glob_last, cc_net):
    ratio_res = []
    for it in range(26):
        cc_class = it
        aux_sum = 0
        aux_other_sum = 0
        layer = 1
        for i in range(pos.shape[1]):
            for j in range(pos.shape[2]):
                if pos[cc_class, i, j] == 1:
                    temp = []
                    last = w_glob_last['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    cc = cc_net[cc_class]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    for p in range(len(cc_net)):
                        temp.append(cc_net[p]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last)
                    temp = np.array(temp)
                    temp = np.delete(temp, cc_class)
                    temp_ave = np.sum(temp) / (len(cc_net) - 1)
                    aux_sum += cc - last
                    aux_other_sum += temp_ave
        if aux_other_sum != 0:
            res = abs(aux_sum) / abs(aux_other_sum)
        else:
            res = 10
        print('label {}-----aux_data:{}, aux_other:{}, ratio:{}'.format(it, aux_sum, aux_other_sum, res))
        ratio_res.append(res)

    # normalize the radio alpha
    ratio_min = np.min(ratio_res)
    ratio_max = np.max(ratio_res)
    for i in range(len(ratio_res)):
        ratio_res[i] = 1.0 + 0.1 * ratio_res[i]
    return ratio_res

def monitoring(cc_net, pos, w_glob_last, w_glob, num_class, num_users, num_samples, args):
    res_monitor = []
    res_monitor_in = []
    for cc_class in range(num_class):
        aux_sum = 0
        aux_other_sum = 0
        glob_sum = 0
        layer = 1
        temp_res = []
        for i in range(pos.shape[1]):
            for j in range(pos.shape[2]):
                if pos[cc_class, i, j] == 1:
                    temp = []
                    last = w_glob_last['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    cc = cc_net[cc_class]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j]
                    for p in range(len(cc_net)):
                        temp.append(cc_net[p]['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last)
                    temp = np.array(temp)
                    temp = np.delete(temp, cc_class)
                    temp_ave = np.sum(temp) / (len(cc_net) - 1)
                    aux_sum += cc - last
                    aux_other_sum += temp_ave


                    glob_temp = (w_glob['resnet.fc{}.weight'.format(layer)].cpu().numpy()[i, j] - last) * num_users * args.local_bs
                    glob_sum += glob_temp
                    res_temp = (glob_temp - num_samples * temp_ave) / (cc - last - temp_ave)
                    # print(res_temp)
                    if 0 < res_temp < num_samples * 1.5 / num_class:
                        temp_res.append(res_temp)
        if len(temp_res) != 0: 
            res_monitor.append(np.mean(temp_res))
        else:
            res_monitor.append(num_samples / num_class)
        if aux_sum - aux_other_sum == 0:
            res = 0
        else:
            res = (glob_sum - num_samples * aux_other_sum) / (aux_sum - aux_other_sum)
        res_monitor_in.append(res)
        
    res_monitor = np.array(res_monitor)
    res_monitor_in = np.array(res_monitor_in)
    return res_monitor, res_monitor_in

def ground_truth_composition(dict_users, idxs_users, num_class, labels):
    res = [0 for i in range(num_class)]
    for idx in idxs_users:
        for i in dict_users[idx]:
            for j in range(num_class):
                temp = np.where(labels[i] == j)
                res[j] += len(temp[0])
    return res

def cosine_similarity(x, y):
    res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return res
