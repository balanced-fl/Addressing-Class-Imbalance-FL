import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from PIL import Image
import math
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

from utils.sampling import EMNIST_client_imbalance, load_EMNIST_data, EMNIST_client_regenerate, ratio_loss_data
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, Net
from models.Fed import FedAvg, outlier_detect, whole_determination
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args.gpu)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'femnist':
        trans_femnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train, label_train, dataset_test, label_test, w_train, w_test = load_EMNIST_data('../emnist-letters.mat', verbose=True, standarized=False)
        if args.iid != 0:
            dict_users = EMNIST_client_imbalance(dataset_train, label_train, w_train, 100, [0, 1, 2, 3, 4, 5, 6], args.ratio)
        else:
            dict_users = EMNIST_client_regenerate(dataset_train, label_train, w_train, 10)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = Net().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    labels = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    ratio = None
    val_acc_list, net_list = [], []
    dict_ratio = ratio_loss_data(dataset_train, label_train, w_train, 26)

    for iter in range(args.epochs):
        if iter > 135:
            args.lr = 0.01
        w_locals, loss_locals, ac_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        np.random.seed(iter)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_users[idx], alpha=ratio, size_average=True)
            w, loss, ac= local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            ac_locals.append(copy.deepcopy(ac))

        # monitor
        if args.loss == 'ratio':
            cc_net, cc_loss = [], []
            aux_class = [i for i in range(26)]
            for i in aux_class:
                cc_local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_ratio[i], alpha=None, size_average=True)
                cc_w, cc_lo, cc_ac = cc_local.train(net=copy.deepcopy(net_glob).to(args.device))
                cc_net.append(copy.deepcopy(cc_w))
                cc_loss.append(copy.deepcopy(cc_lo))
            pos = outlier_detect(w_glob, cc_net, iter)

        # labeling process and updating global model
        w_glob_last = copy.deepcopy(w_glob)
        w_glob = FedAvg(w_locals)

        if args.loss == 'ratio':
            ratio = torch.tensor(whole_determination(pos, w_glob_last, cc_net), dtype=torch.float)
            print(ratio)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        ac_avg = sum(ac_locals) / len(ac_locals)
        print('Round {:3d}, Average loss {:.3f}, Accuracy {:.3f}\n'.format(iter, loss_avg, ac_avg))
        loss_train.append(loss_avg)

    # testing
    net_glob.eval()
    print('FL(femnist): {}, {}:1, former 7 classes, {} eps, {} local eps'.format(args.loss, int(1 / args.ratio), args.epochs, args.local_ep))

    acc_test, loss_test = test_img(net_glob, dataset_test, label_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))

