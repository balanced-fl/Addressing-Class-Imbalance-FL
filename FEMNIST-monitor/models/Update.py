import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, args, class_num, alpha=None, gamma=2, size_average=True):
        self.args = args
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        alpha = alpha.to(self.args.device)
        probs = probs.to(self.args.device)
        log_p = log_p.to(self.args.device)

        batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Ratio_Cross_Entropy(nn.Module):
    r"""
            This criterion is a implemenation of Ratio Loss, which is proposed to solve the imbalanced
            problem in Fderated Learning

                Loss(x, class) = - \alpha  \log(softmax(x)[class])

            The losses are averaged across observations for each minibatch.

            Args:
                alpha(1D Tensor, Variable) : the scalar factor for this criterion
                size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                    However, if the field size_average is set to False, the losses are
                                    instead summed for each minibatch.
        """

    def __init__(self, args, class_num, alpha=None, size_average=False):
        self.args = args
        super(Ratio_Cross_Entropy, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        # print(inputs)
        # print(P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        alpha = alpha.to(self.args.device)
        probs = probs.to(self.args.device)
        log_p = log_p.to(self.args.device)

        batch_loss = - alpha * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=30,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        edges = self.edges
        mmt = self.momentum
        

        N = pred.size(0)
        C = pred.size(1)
        P = F.softmax(pred)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)
        weights = torch.zeros_like(g)

        # valid = label_weight > 0
        # tot = max(valid.float().sum().item(), 1.0)
        tot = pred.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) 
            num_in_bin = inds.sum().item()
            # print(num_in_bin)
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        
        # print(pred)
        # pred = P * weights
        # print(pred)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print(probs)
        # print(probs)
        # print(log_p.size(), weights.size())

        batch_loss = -log_p * weights / tot
        # print(batch_loss)
        loss = batch_loss.sum()
        # print(loss)
        return loss



class DatasetSplit(Dataset):
    def __init__(self, dataset, labels, idxs):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if len(self.labels) != 0:
            image = self.dataset[self.idxs[item]]
            label = self.labels[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = np.argmax(pred, 1)
    count = 0
    for i in range(len(test_np)):
        if test_np[i] == label[i]:
            count += 1
    return np.sum(count), len(test_np)

class LocalUpdate(object):
    def __init__(self, args, dataset=None, label=None, idxs=None, alpha=None, size_average=False):
        self.args = args
        if self.args.loss == 'mse':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif self.args.loss == 'focal':
            self.loss_func = FocalLoss(class_num=26, alpha=alpha, args=args, size_average=size_average)
        elif self.args.loss == 'ratio' or self.args.loss == 'ce':
            self.loss_func = Ratio_Cross_Entropy(class_num=26, alpha=alpha, size_average=size_average, args=args)
        elif self.args.loss == 'ghm':
            self.loss_func = GHMC()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, label, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        epoch_loss = []
        epoch_ac = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_ac = []
            batch_whole = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(images, labels)
                images = images.unsqueeze(1)
                # labels = labels.unsqueeze(0)
                images, labels = images.to(self.args.device, dtype=torch.float), labels.to(self.args.device, dtype=torch.long)
                # images = images.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                ac, whole = AccuarcyCompute(log_probs, labels)
                # ac, whole = 0, 0

                # ghm loss
                '''
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                target_one_hot = label_binarize(labels.data.view_as(y_pred).cpu().numpy(), [i for i in range(26)])
                target_one_hot = torch.from_numpy(target_one_hot)
                target_one_hot = target_one_hot.to(self.args.device)
                loss = self.loss_func(log_probs, target_one_hot, labels)'''


                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, batch_idx * len(images), len(self.ldr_train.dataset),100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_ac.append(ac)
                batch_whole.append(whole)
            epoch_ac.append(sum(batch_ac) / sum(batch_whole))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_ac) / len(epoch_ac)
