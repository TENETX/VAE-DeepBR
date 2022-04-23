# -*- coding: UTF-8 -*-
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from utills.ml_metrics import all_metrics
from Vutils import gauss_kl_loss
import numpy as np
from Vmain import label_enhance
# from torch.optim import lr_scheduler


def train(model, device, views_data_loader, args, loss_coefficient,
          train_features, train_partial_labels, test_features, test_labels, choose, enc=None, dec=None, fold=1):

    opti = []

    # init optimizer
    if args.opt == 'adam':
        if choose == 2:
            opti_all = optim.Adam(list(model.parameters(
            )) + list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=1e-5)
            opti = [opti_all]
        elif choose == 3:
            opti_ml = optim.Adam(list(model.parameters()),
                                 lr=args.lr, weight_decay=1e-5)
            opti_le = optim.Adam(list(enc.parameters(
            )) + list(dec.parameters()), lr=args.Vlearning_rate, weight_decay=1e-5)
            opti = [opti_ml, opti_le]
        else:
            opti_br = optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-5)
            opti = [opti_br]
        # scheduler = lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[500, 800, 900], gamma=0.2, last_epoch=-1)
        # scheduler.step()
    else:
        # SGD很少用
        return
    # train model
    trainer = Trainer(model, views_data_loader, args.epoch, opti, args.show_epoch,
                      loss_coefficient, args.model_save_epoch, args.model_save_dir, device, choose, args.smoothing_rate, enc, dec)
    loss_list = trainer.fit(
        fold, train_features, train_partial_labels, test_features, test_labels, args)

    return loss_list


@ torch.no_grad()
def test(model, features, labels, device, model_state_path=None, is_eval=False, args=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))

    metrics_results = None
    model.eval()

    # CUDA
    for i, _ in enumerate(features):
        features[i] = features[i].to(device)
    labels = labels.to(device)
    # prediction
    with torch.no_grad():
        outputs = model(features)

    outputs = outputs.cpu().numpy()
    preds = (outputs > 0.5).astype(int)

    # eval
    if is_eval:
        target = labels.cpu().numpy()
        target = target.astype(np.uint8)
        metrics_results = all_metrics(outputs, preds, target)

    return metrics_results, preds


class Trainer(object):
    def __init__(self, model, train_data_loader, epoch, optimizer, show_epoch,
                 loss_coefficient, model_save_epoch, model_save_dir, device, choose, smoothing, enc=None, dec=None):
        self.model = model
        self.enc = enc
        self.dec = dec
        self.train_data_loader = train_data_loader
        self.epoch = epoch
        if choose == 1:
            self.opti_br = optimizer[0]
        elif choose == 2:
            self.opti_all = optimizer[0]
        else:
            self.opti_ml = optimizer[0]
            self.opti_le = optimizer[1]
        self.show_epoch = show_epoch
        self.loss_coefficient = loss_coefficient
        self.model_save_epoch = model_save_epoch
        self.model_save_dir = model_save_dir
        self.device = device
        self.choose = choose
        self.smoothing = smoothing

    def fit(self, fold, train_features, train_partial_labels, test_features, test_labels, args=None):
        loss_list = []

        # VAE records
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 为了加快速度，索性在外面套判断
        if self.choose == 1:
            for epoch in range(self.epoch):
                self.model.train()
                for step, train_data in enumerate(self.train_data_loader):
                    inputs, labels, index = train_data
                    # print(inputs)
                    outputs = self.model(inputs)
                    # BR loss
                    with torch.no_grad():
                        labels = labels * (1 - self.smoothing) + \
                            0.5 * self.smoothing
                    lossBR = F.binary_cross_entropy(outputs, labels)
                    # print_str = f'Epoch: {epoch}\t BR Loss: {lossBR:.4f} \t'
                    # show loss info
                    if epoch % self.show_epoch == 0 and step == 0:
                        epoch_loss = dict()
                        loss_list.append(epoch_loss)
                        # print(print_str)
                    self.opti_br.zero_grad()
                    lossBR.backward()
                    self.opti_br.step()
        elif self.choose == 2:
            for epoch in range(self.epoch):
                self.enc.train()
                self.dec.train()
                self.model.train()
                for step, train_data in enumerate(self.train_data_loader):
                    inputs, labels, index = train_data
                    batch_x = inputs[0]
                    batch_y = labels
                    batch_x, batch_y = batch_x.to(
                        self.device), batch_y.to(self.device)
                    batch_data = torch.cat((batch_x, batch_y), 1)

                    # forward
                    (mu, sigma) = self.enc(batch_data)
                    z = mu + sigma * (torch.randn(mu.size()).to(self.device))
                    batch_x_hat = self.dec(z)
                    d = F.sigmoid(z[:, -args.Vnum_class:])

                    # VAE loss
                    rec_loss_x = F.mse_loss(batch_x_hat, batch_x)
                    kl_loss = gauss_kl_loss(mu, sigma)
                    rec_loss_y = F.binary_cross_entropy(d, batch_y)
                    lossVAE = rec_loss_y + args.Valpha * kl_loss + args.Vbeta * rec_loss_x

                    # print(inputs)
                    outputs = self.model(inputs)
                    labels = d.detach().to(self.device)

                    # BR loss
                    with torch.no_grad():
                        labels = labels * (1 - self.smoothing) + \
                            0.5 * self.smoothing
                    lossBR = F.binary_cross_entropy(outputs, labels)
                    loss = lossBR + lossVAE
                    # print_str = f'Epoch: {epoch}\t loss: {loss:.4f} \t VAE Loss: {lossVAE:.4f} \t BR Loss: {lossBR:.4f} \t'
                    # show loss info
                    if epoch % self.show_epoch == 0 and step == 0:
                        epoch_loss = dict()
                        # writer.add_scalar("Loss/train", lossBR, epoch)  # log
                        # plotter.plot('loss', 'train', 'Class Loss', epoch, _ML_loss)
                        loss_list.append(epoch_loss)
                        # print(print_str)
                    self.opti_all.zero_grad()
                    loss.backward()
                    self.opti_all.step()
        else:
            for epoch in range(self.epoch):
                self.enc.train()
                self.dec.train()
                for step, train_data in enumerate(self.train_data_loader):
                    inputs, labels, index = train_data
                    batch_x = inputs[0]
                    batch_y = labels
                    batch_x, batch_y = batch_x.to(
                        self.device), batch_y.to(self.device)
                    batch_data = torch.cat((batch_x, batch_y), 1)
                    # forward
                    (mu, sigma) = self.enc(batch_data)
                    z = mu + sigma * (torch.randn(mu.size()).to(self.device))
                    batch_x_hat = self.dec(z)
                    d = F.sigmoid(z[:, -args.Vnum_class:])
                    # VAE loss
                    rec_loss_x = F.mse_loss(batch_x_hat, batch_x)
                    kl_loss = gauss_kl_loss(mu, sigma)
                    rec_loss_y = F.binary_cross_entropy(d, batch_y)
                    lossVAE = rec_loss_y + args.Valpha * kl_loss + args.Vbeta * rec_loss_x
                    # print_str = f'Epoch: {epoch}\t loss: {loss:.4f} \t VAE Loss: {lossVAE:.4f} \t BR Loss: {lossBR:.4f} \t'
                    # show loss info
                    if epoch % self.show_epoch == 0 and step == 0:
                        epoch_loss = dict()
                        # writer.add_scalar("Loss/train", lossBR, epoch)  # log
                        # plotter.plot('loss', 'train', 'Class Loss', epoch, _ML_loss)
                        loss_list.append(epoch_loss)
                        # print(print_str)
                    self.opti_le.zero_grad()
                    lossVAE.backward()
                    self.opti_le.step()
            for epoch in range(self.epoch):
                self.model.train()
                for step, train_data in enumerate(self.train_data_loader):
                    inputs, labels, index = train_data
                    # label_enhance
                    labels = label_enhance(self.enc, inputs[0], labels.numpy(), args)
                    labels = torch.from_numpy(labels).to(self.device)
                    outputs = self.model(inputs)
                    # BR loss
                    with torch.no_grad():
                        labels = labels * (1 - self.smoothing) + \
                            0.5 * self.smoothing
                    lossBR = F.binary_cross_entropy(outputs, labels)
                    # print_str = f'Epoch: {epoch}\t BR Loss: {lossBR:.4f} \t'
                    # show loss info
                    if epoch % self.show_epoch == 0 and step == 0:
                        epoch_loss = dict()
                        loss_list.append(epoch_loss)
                        # print(print_str)
                    self.opti_ml.zero_grad()
                    lossBR.backward()
                    self.opti_ml.step()
        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
