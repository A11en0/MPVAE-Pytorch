# -*- coding: UTF-8 -*-
import os
from functools import reduce
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import build_graph, estimating_label_correlation_matrix, label_propagation
from mpvae import compute_loss
from utils.ml_metrics import all_metrics


@ torch.no_grad()
def test(model, features, labels, weight_var, device, model_state_path=None, is_eval=False):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))

    metrics_results = None

    model.eval()
    
    # CUDA
    for i, _ in enumerate(features):
        features[i] = features[i].to(device)
    labels = labels.to(device)

    # prediction
    input_feat = torch.cat(list(features.values()), dim=1)
    input_label = labels
    with torch.no_grad():
        label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(input_label, input_feat)

    outputs = feat_out.cpu().numpy()
    preds = (outputs > 0.5).astype(int)

    # eval
    if is_eval:
        target = labels.int().cpu().numpy()
        metrics_results = all_metrics(outputs, preds, target)

    return metrics_results, preds

class Trainer(object):
    def __init__(self, model, args, device):
        self.model = model
        self.epochs = args.epochs
        self.show_epoch = args.show_epoch
        self.model_save_epoch = args.model_save_epoch
        self.model_save_dir = args.model_save_dir
        self.device = device
        self.args = args

        if args.opt == 'adam':
            self.opti = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.opti = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.lr_s = torch.optim.lr_scheduler.StepLR(self.opti, step_size=10, gamma=0.9)

    def fit(self, train_loader, train_features, train_partial_labels, test_features, test_labels, fold):
        loss_list = []
        best_F1, best_epoch = 0.0, 0.0
        train_partial_labels_np, train_pred_np, train_pred_np, train_lp_np, labels_lp = [], [], [], [], []
        Wn, L = [], []

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if self.args.using_lp:
            train_partial_labels_np = train_partial_labels.numpy().copy()
            train_pred_np = train_partial_labels_np.copy()
            train_lp_np = train_partial_labels_np.copy()

            Wn = 0.0
            for id, view_feature in train_features.items():
                view_feature = view_feature.numpy()
                Wn_tmp = build_graph(view_feature, k=self.args.neighbors_num, args=self.args)
                Wn += Wn_tmp

            L = estimating_label_correlation_matrix(train_partial_labels_np)

        for epoch in range(self.epochs):
            self.model.train()

            if self.args.using_lp:
                maxiter = self.args.maxiter
                train_lp_np = label_propagation(self.args, Wn, L, train_pred_np, train_partial_labels_np,
                                              train_lp_np, gamma=self.args.gamma, alpha=self.args.alpha,
                                              zeta=self.args.zeta, maxiter=maxiter)

            for step, (inputs, labels, index) in enumerate(train_loader):
                # adjust_learning_rate(epoch, self.args, self.opti)

                if self.args.using_lp:
                    labels_lp = torch.from_numpy(train_lp_np[index]).float().to(self.device)

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)

                input_feat = torch.cat(list(inputs.values()), dim=1)
                input_label = labels.to(self.device)
                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = self.model(input_label, input_feat)

                # train the model for one step and log the training loss
                if self.args.residue_sigma == "random":
                    r_sqrt_sigma = torch.from_numpy(np.random.uniform(-np.sqrt(6.0 / (self.args.label_dim + self.args.z_dim)),
                                                                      np.sqrt(6.0 / (self.args.label_dim + self.args.z_dim)),
                                                                      (self.args.label_dim, self.args.z_dim))).to(self.device)
                    loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label,
                                                                                                           label_out,
                                                                                                           label_mu,
                                                                                                           label_logvar,
                                                                                                           feat_out,
                                                                                                           feat_mu,
                                                                                                           feat_logvar,
                                                                                                           r_sqrt_sigma,
                                                                                                           self.device,
                                                                                                           self.args)
                else:
                    loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label,
                                                                                                           label_out,
                                                                                                           label_mu,
                                                                                                           label_logvar,
                                                                                                           feat_out,
                                                                                                           feat_mu,
                                                                                                           feat_logvar,
                                                                                                           self.model.r_sqrt_sigma,
                                                                                                           self.device,
                                                                                                           self.args)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}'

                # show loss info
                if epoch % self.show_epoch == 0 and step == 0:
                    print(print_str)
                    epoch_loss = dict()
                    # writer.add_scalar("Loss/train", loss, epoch)  # log
                    # plotter.plot('loss', 'train', 'Class Loss', epoch, _ML_loss)
                    loss_list.append(epoch_loss)

                self.opti.zero_grad()
                loss.backward()
                self.opti.step()
                # self.lr_s.step()

                if self.args.using_lp:
                    _, train_pred_np = test(self.model, train_features, train_partial_labels, None,
                                            self.device, is_eval=False)

            # evaluation
            if epoch % self.show_epoch == 0 and self.args.is_test_in_train:
                metrics_results, _ = test(self.model, test_features, test_labels, None,
                                          self.device, is_eval=True)

                # draw figure to find best epoch number
                for i, key in enumerate(metrics_results):
                    print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                    loss_list[epoch][i] = metrics_results[key]
                print("\n")

                if best_F1 < metrics_results['micro_f1']:
                    best_F1, best_epoch = metrics_results['micro_f1'], epoch

        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)