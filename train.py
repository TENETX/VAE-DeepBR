import math
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from utills.ml_metrics import all_metrics
from Vutils import gauss_kl_loss
import numpy as np
# from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='scalar')


def train(model, device, data_loader, args, loss_coefficient,
          train_features, train_partial_labels, test_features, test_labels, enc=None, dec=None, fold=1):
    # init optimizer
    if args.opt == 'adam':
        opti = optim.Adam(list(model.parameters(
        )) + list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=1e-5)
    else:
        # SGD很少用
        return
    # train model
    trainer = Trainer(model, data_loader, args.epoch, opti, args.show_epoch,
                      loss_coefficient, args.model_save_epoch, args.model_save_dir, device, args.smoothing_rate, enc, dec)
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
                 loss_coefficient, model_save_epoch, model_save_dir, device, smoothing, enc=None, dec=None):
        self.model = model
        self.enc = enc
        self.dec = dec
        self.train_data_loader = train_data_loader
        self.epoch = epoch
        self.opti_all = optimizer
        self.show_epoch = show_epoch
        self.loss_coefficient = loss_coefficient
        self.model_save_epoch = model_save_epoch
        self.model_save_dir = model_save_dir
        self.device = device
        self.smoothing = smoothing

    def fit(self, fold, train_features, train_partial_labels, test_features, test_labels, args=None):
        loss_list = []
        # VAE records
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        for epoch in range(self.epoch):
            self.enc.train()
            self.dec.train()
            self.model.train()
            num_x = 0
            for step, train_data in enumerate(self.train_data_loader):
                inputs, labels_r, index = train_data
                batch_x = inputs[0]
                batch_y = labels_r
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
                # 需要选择合适的loss函数
                # rate = 1
                # rate = np.exp(-epoch)
                # rate = math.cos((epoch / self.epoch) * math.pi / 2)
                if epoch <= 100:
                    rate = 0
                elif epoch > 100:
                    rate = 0.2
                elif epoch > 200:
                    rate = 0.4
                elif epoch > 300:
                    rate = 0.6
                elif epoch > 400:
                    rate = 0.8
                else:
                    rate = 0.9
                # ------------------------------------------
                labels_new = labels_r * rate + labels * (1 - rate)
                # BR loss
                newM = (labels_new * torch.log(outputs) + (1.0 - labels_new) * torch.log(1.0 - outputs))
                lossBR = - newM.mean()
                loss = lossBR + lossVAE
                num_x += 1
                loss_list.append(loss.item())
                # print_str = f'Epoch: {epoch}\t loss: {loss:.4f} \t VAE Loss: {lossVAE:.4f} \t BR Loss: {lossBR:.4f} \t'
                # show loss info
                self.opti_all.zero_grad()
                loss.backward()
                self.opti_all.step()
                if epoch % self.show_epoch == 0 and step == 0:
                    epoch_loss = dict()
                    loss_list.append(epoch_loss)
                    metrics_results, _ = test(
                        self.model, test_features, test_labels, self.device, is_eval=True, args=args)
                    writer.add_scalar("Smooth/Smooth", rate, epoch)
                    writer.add_scalar("Loss/lossVAE", lossVAE, epoch)
                    writer.add_scalar("Loss/lossBR", lossBR, epoch)
                    writer.add_scalar("Loss/loss", loss, epoch)
                    writer.add_scalar(
                        "hamming_loss", metrics_results["hamming_loss"], epoch)
                    writer.add_scalar(
                        "ranking_loss", metrics_results["ranking_loss"], epoch)
                    writer.add_scalar(
                        "avg_precision", metrics_results["avg_precision"], epoch)
                    writer.add_scalar("Loss/lossVAE", lossVAE, epoch)
                    writer.add_scalar("Loss/lossBR", lossBR, epoch)
                    writer.add_scalar("Loss/loss", loss, epoch)
                    writer.add_scalar(
                        "hamming_loss", metrics_results["hamming_loss"], epoch)
                    writer.add_scalar(
                        "ranking_loss", metrics_results["ranking_loss"], epoch)
                    writer.add_scalar(
                        "avg_precision", metrics_results["avg_precision"], epoch)
        writer.close()
        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
