import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from sklearn.preprocessing import label_binarize

class datasetsplit(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.dataset[item]
        label = self.labels[item]

        return image, label


def test_img(net_g, datatest, label, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    auc_final = []
    auc_final_new = []
    data_loader = DataLoader(datasetsplit(datatest, label), batch_size=args.bs)
    l = len(data_loader)
    false = np.zeros((26, 1))
    false_all = np.zeros((26, 1))
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # gpu acceleration
            data, target = data.cuda(), target.cuda()
        data = data.unsqueeze(1)
        data, target = data.to(args.device, dtype=torch.float), target.to(args.device, dtype=torch.long)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        #print(y_pred.eq(target.data.view_as(y_pred)).long().cpu())

        pred = log_probs.data.cpu().numpy()
        pred_one_hot = label_binarize(y_pred.data.cpu().numpy(), [i for i in range(26)])
        target_one_hot = label_binarize(target.data.view_as(y_pred).cpu().numpy(), [i for i in range(26)])
        auc_temp = metrics.roc_auc_score(target_one_hot, pred, average='micro')
        auc_temp_new = metrics.roc_auc_score(target_one_hot, pred_one_hot, average='micro')
        auc_final.append(auc_temp)
        auc_final_new.append(auc_temp_new)
        for i in range(torch.numel(y_pred)):
            false_all[target.data.view_as(y_pred)[i][0], 0] += 1
            if y_pred.eq(target.data.view_as(y_pred)).long().cpu()[i][0] == 0:
                false[target.data.view_as(y_pred)[i][0], 0] += 1

    test_loss /= len(data_loader.dataset)
    auc_res = np.mean(auc_final)
    auc_res_new = np.mean(auc_final_new)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    for i in range(len(false)):
        ac_temp = (false_all[i, 0] - false[i, 0]) / false_all[i, 0]
        print('{:.4f}'.format(ac_temp))
    print('AUC Score: {:.6f}, AUC Score New: {:.6f}'.format(auc_res, auc_res_new))
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\nAUC Score: {:.4f}'.format(
            test_loss, correct, len(data_loader.dataset), accuracy, auc_res))
    return accuracy, test_loss
