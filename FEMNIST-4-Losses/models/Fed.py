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
            
            if w[max_index, i, j] == 0:
                outlier = np.where(temp == w[max_index, i, j])
            else:
                outlier = np.where(np.abs(temp) / abs(w[max_index, i, j]) > 0.80)
            if len(outlier[0]) < 2:
                pos_res[max_index, i, j] = 1
            # pos_res[max_index, i, j] = 1
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
        # add a upper bound to the ratio
        if ratio_res[i] >= 5000:
            ratio_res[i] = 5000
        ratio_res[i] = 1.0 + 0.1 * ratio_res[i]
        # ratio_res[i] = 1.5 - 0.3  * (ratio_res[i] - ratio_min) / (ratio_max - ratio_min)
    return ratio_res

