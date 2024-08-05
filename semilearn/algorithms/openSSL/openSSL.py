
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_optimizer, get_cosine_schedule_with_warmup
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from semilearn.algorithms.openSSL.utils import single_kmeans, update_semantic_prototypes, \
     get_cluster_labels, get_shadow_centroids, get_domain_cluster_adaptive
from easydl import GradientReverseModule, aToBSheduler
from sklearn.metrics import accuracy_score, confusion_matrix
from semilearn.core.criterions import IBLoss, SupConLoss, entropyLoss, vaeLoss

class openSSLNet(nn.Module):
    def __init__(self, base, proj_size=128):
        super(openSSLNet, self).__init__()
        self.backbone = base
        self.num_features = base.num_features

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])

        self.channels = base.channels
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
        self.fc_mu = nn.Linear(self.channels, self.channels)
        self.fc_logvar = nn.Linear(self.channels, self.channels)

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x,  **kwargs):
        feat = self.backbone(x, only_feat=True)
        mu, logvar = self.fc_mu(feat), self.fc_logvar(feat)
        feat= self.reparameterize(mu, logvar)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits': logits, 'feat': feat, 'feat_proj': feat_proj, 'mu':mu, 'logvar':logvar}

    # def forward(self, x, stats=False,  **kwargs):
    #     feat = self.backbone(x, only_feat=True)
    #     logits = self.backbone(feat, only_fc=True)
    #     feat_proj = self.l2norm(self.mlp_proj(feat))
    #     if stats:
    #         mu, logvar = self.fc_mu(feat), self.fc_logvar(feat)
    #         feat_reparam = self.reparameterize(mu, logvar)
    #         return {'logits': logits, 'feat': feat, 'feat_proj': feat_proj, 'mu':mu, 'logvar':logvar, 'feat_reparam': feat_reparam}
    #     return {'logits': logits, 'feat': feat, 'feat_proj': feat_proj}


    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def disentangle(self, x, reverse=True):
        # using gradiant reveral layer: use entropy minimization loss
        # else, use negative entropy
        if reverse:
            x = self.grl(x)
        # x = self.classifier(x.detach())
        x = self.backbone(x.detach(), only_fc=True)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dims = [256, 128, 64, 32], input_dim=1024, img_size=32):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.about_img_size=img_size//32
        self.decoder_input = nn.Linear(input_dim, hidden_dims[0] * 4 * self.about_img_size* self.about_img_size)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1),
                            nn.Sigmoid())


    def forward(self, x):
        assert x.size(1) == self.input_dim

        out = self.decoder_input(x)
        out = out.view(-1, self.hidden_dims[0], 2 * self.about_img_size, 2 * self.about_img_size)
        # out = out.view(-1, 256, 2* self.about_img_size, 2* self.about_img_size)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out


@ALGORITHMS.register('openSSL')
class openSSL(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        self.lambda_c = args.cluster_loss_ratio
        self.T = args.T
        self.p_cutoff = args.p_cutoff
        self.smoothing_alpha = args.smoothing_alpha
        self.use_da = args.use_da
        self.da_len = args.da_len
        self.lambda_p = 1.0

        self.len_lb = len(self.dataset_dict['train_lb'])
        self.len_ulb = len(self.dataset_dict['train_ulb'])
        self.cluster_interval = self.len_ulb // (args.all_batch_size * args.uratio) + 1

        self.cluster_scale = args.cluster_scale
        self.min_cluster = args.min_cluster
        self.num_centroids = self.num_classes * self.cluster_scale
        self.structural_centroids = None
        self.shadow_centroids = None
        self.structural_assignments = None

        self.feats_lb = torch.zeros(self.len_lb, self.model['model_c'].num_features).cuda(self.gpu)
        self.feats_proj_lb = F.normalize(torch.zeros(self.len_lb, self.args.proj_size).cuda(self.gpu), 1)
        self.labels_lb = torch.zeros(self.len_lb, dtype=torch.long).cuda(self.gpu)
        self.plabels_ulb = torch.zeros(self.len_ulb, dtype=torch.long).cuda(self.gpu) - 1
        self.semantic_prototypes = None

        self.feats_ulb = torch.zeros(self.len_ulb, self.model['model_c'].num_features).cuda(self.gpu)
        self.feats_proj_ulb = F.normalize(torch.zeros(self.len_ulb, self.args.proj_size).cuda(self.gpu), 1)
        self.cluster_labels = None

        self.use_ema_bank = args.use_ema_bank
        self.ema_bank_m = args.ema_bank_m

        # self.best_out_acc, self.best_out_it = 0.0, 0
        # self.best_all_acc, self.best_all_it = 0.0, 0


        self.IB_Loss = IBLoss()
        self.lambda_ib = args.lambda_ib

        self.contrastive_loss = SupConLoss(args, contrast_mode='one')
        self.lambda_con = args.lambda_con

        self.disentangle = args.disentangle
        self.lambda_disent = args.lambda_disent

        self.entropy_loss = entropyLoss()
        self.rec_criterion = nn.SmoothL1Loss()

        self.vae_loss = vaeLoss()
        self.lambda_vae = args.lambda_vae


        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class)


    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class


    def set_hooks(self):
        # self.register_hook(
        #     DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
        #     "DistAlignHook")
        # self.register_hook(FixedThresholdingHook(), "MaskingHook")


        from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
        from .utils import SoftMatchWeightingHook
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p,
                             p_target_type='uniform' if self.args.dist_uniform else 'model'),
            "DistAlignHook")
        self.register_hook(
            SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p,
                                   per_class=self.args.per_class), "MaskingHook")


        from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, \
            DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook, openSSL_EMAHook, \
            openSSL_TimerHook, openSSL_ParamUpdateHook

        self.register_hook(openSSL_ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(openSSL_EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(openSSL_TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    @torch.no_grad()
    def update_bank(self, feats, feats_proj, labels, indices, labeled=True):
        if self.distributed and self.world_size > 1:
            feats = concat_all_gather(feats)
            feats_proj = concat_all_gather(feats_proj)
            labels = concat_all_gather(labels)
            indices = concat_all_gather(indices)

        if labeled:
            self.feats_lb[indices] = feats
            self.feats_proj_lb[indices] = feats_proj
            self.labels_lb[indices] = labels
        else:
            self.feats_ulb[indices] = feats
            self.feats_proj_ulb[indices] = feats_proj
            self.plabels_ulb[indices] = labels


    def set_domain_cluster(self, dataloader):
        self.args.cluster_dir = os.path.join(self.args.cluster_dir, self.args.cluster_method)
        cluster_dir = self.args.cluster_dir
        os.makedirs(cluster_dir,exist_ok=True)
        # cluster_label_path = os.path.join(cluster_dir, 'cluster_label.bin')

        # cluster_label_path = os.path.join(cluster_dir, 'cluster_label_cluster5.npy')
        cluster_label_path = os.path.join(cluster_dir, self.args.cluster_label_file)
        if os.path.exists(cluster_label_path):
            # cluster_label = np.fromfile(cluster_label_path, dtype=np.int64)
            cluster_label = np.load(cluster_label_path)
            # setattr(self.args, 'num_domains', 5)
            self.num_domains = 5

        else:
            best_clusters, cluster_label = get_domain_cluster_adaptive(self.args, dataloader, self.print_fn)
            self.num_domains = best_clusters

        self.print_fn('finish clustering')
        self.loader_dict['train_ulb'].dataset.set_cluster(np.array(cluster_label))


    def set_model(self):
        # domain cluster
        self.set_domain_cluster(self.loader_dict['train_ulb'])

        model_c = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain,
                                 pretrained_path=self.args.pretrain_path)
        model_s = self.net_builder(num_classes=self.num_domains, pretrained=self.args.use_pretrain,
                                 pretrained_path=self.args.pretrain_path)

        decoder = Decoder(input_dim=self.args.feat_dim, img_size=self.args.img_size)

        model_c = openSSLNet(model_c, proj_size=self.args.proj_size)
        model_s = openSSLNet(model_s, proj_size=self.args.proj_size)
        return {'model_c': model_c, 'model_s': model_s, 'decoder': decoder}


    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = openSSLNet(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model['model_c'].state_dict()))
        return ema_model

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer_c = get_optimizer(self.model['model_c'], self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
                                  self.args.layer_decay)
        optimizer_s = get_optimizer(self.model['model_s'], self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
                                  self.args.layer_decay)

        optimizer_decoder = get_optimizer(self.model['decoder'], self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
                                  self.args.layer_decay)

        scheduler_c = get_cosine_schedule_with_warmup(optimizer_c,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        scheduler_s = get_cosine_schedule_with_warmup(optimizer_s,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        scheduler_decoder = get_cosine_schedule_with_warmup(optimizer_decoder,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)

        optimizer = {'optimizer_c': optimizer_c, 'optimizer_s':optimizer_s, 'optimizer_decoder':optimizer_decoder}
        scheduler = {'scheduler_c': scheduler_c, 'scheduler_s':scheduler_s, 'scheduler_decoder':scheduler_decoder}

        return optimizer, scheduler

    def train_step(self, idx_lb, idx_ulb, x_lb, y_lb, x_ulb_w, x_ulb_s, cluster_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = idx_ulb.shape[0]
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs_c = self.model['model_c'](inputs)
                logits_c, content, content_proj, mu, logvar = outputs_c['logits'], outputs_c['feat'], outputs_c['feat_proj'], outputs_c['mu'], outputs_c['logvar']
                logits_c_x_lb, content_x_lb, content_proj_x_lb = logits_c[:num_lb], content[:num_lb], content_proj[:num_lb]
                logits_c_x_ulb_w, logits_c_x_ulb_s = logits_c[num_lb:].chunk(2)
                content_x_ulb_w,_ = content[num_lb:].chunk(2)
                content_proj_x_ulb_w, content_proj_x_ulb_s = content_proj[num_lb:].chunk(2)


                outputs_s = self.model['model_s'](inputs)
                logits_s, style, style_proj = outputs_s['logits'], outputs_s['feat'], outputs_s['feat_proj']
                logits_s_x_lb, style_x_lb, style_proj_x_lb = logits_s[:num_lb], style[:num_lb], style_proj[:num_lb]

                logits_s_x_ulb_w, _ = logits_s[num_lb:].chunk(2)
                style_x_ulb_w, style_x_ulb_s = style[num_lb:].chunk(2)
                style_proj_x_ulb_w, style_proj_x_ulb_s = style_proj[num_lb:].chunk(2)

            else:
                raise ValueError("Must set use_cat as True!")

            # 求标记样本的特征prototypes
            if self.it > 0 and self.it % (self.len_lb // self.args.all_batch_size + 1) == 0:
                self.semantic_prototypes = update_semantic_prototypes(self.feats_proj_lb, self.labels_lb,
                                                                      self.num_classes)

            if self.it > 0 and self.it % self.cluster_interval == 0:
                gpu_kmeans = False
                if self.args.dataset in ['domainnet', 'domainnet_balanced']:
                    gpu_kmeans = True
                self.structural_centroids, self.structural_assignments = single_kmeans(self.num_centroids,
                                                                                       self.feats_proj_ulb,
                                                                                       seed=self.args.seed,
                                                                                       use_gpu=gpu_kmeans)
                self.shadow_centroids = get_shadow_centroids(self.structural_assignments, self.num_centroids,
                                                             self.feats_ulb)
                self.cluster_labels = get_cluster_labels(self.semantic_prototypes, self.structural_centroids,
                                                         min_cluster=self.min_cluster)

            # supervised loss
            sup_loss = self.ce_loss(logits_c_x_lb, y_lb, reduction='mean')


            with torch.no_grad():
                logits_c_x_ulb_w = logits_c_x_ulb_w.detach()
                content_proj_x_ulb_w = content_proj_x_ulb_w.detach()

                probs_sem = self.compute_prob(logits_c_x_ulb_w)
                if self.use_da:
                    # probs_sem = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_sem.detach())

                    probs_x_lb = self.compute_prob(logits_c_x_lb.detach())
                    # uniform distribution alignment
                    probs_sem = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_sem, probs_x_lb=probs_x_lb)




            if self.epoch > 0 and self.smoothing_alpha < 1 and self.structural_centroids is not None:
                sim = torch.exp(torch.mm(content_proj_x_ulb_w, self.structural_centroids.t()) / self.T)
                sim_probs = sim / sim.sum(1, keepdim=True)
                probs_clus = torch.zeros((num_ulb, self.num_classes)).cuda(self.gpu)
                for c in range(self.num_classes):
                    mask_c = (self.cluster_labels == c)
                    probs_clus[:, c] = sim_probs[:, mask_c].sum(1)
                probs_smoothed = probs_sem * self.smoothing_alpha + probs_clus * (1 - self.smoothing_alpha)
                probs = probs_smoothed
            else:
                probs = probs_sem  # 未标记样本的logits

            if self.epoch > 0 and self.structural_centroids is not None:
                with torch.no_grad():
                    ctr_prt_sim = torch.exp(torch.mm(self.structural_centroids, self.semantic_prototypes.t()) / self.T)
                    p_centroids = ctr_prt_sim / ctr_prt_sim.sum(1, keepdim=True)
                    W_local = F.one_hot(self.structural_assignments[idx_ulb], num_classes=self.num_centroids)
                    ins_ctr_sim = torch.mm(probs_sem, p_centroids.t())
                    W_global = ins_ctr_sim / ins_ctr_sim.sum(1, keepdim=True)
                    W_glocal = W_global + W_local
                    W_glocal = W_glocal / W_glocal.sum(1, keepdim=True)
                cluster_loss = self.cluster_loss(content_proj_x_ulb_s, W_glocal, self.T)
            else:
                cluster_loss = torch.zeros(1).cuda(self.gpu)

            # pl_ulb = probs.max(1)[1]
            # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False)
            # pl_ulb[~(mask.bool())] = -1
            # unsup_loss = self.consistency_loss(logits_c_x_ulb_s,
            #                                    probs,
            #                                    'ce',
            #                                    mask=mask)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs)
            # generate unlabeled targets using pseudo label hook
            pl_ulb = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pesudo labels
                                          logits=logits_c_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            # calculate loss
            unsup_loss = self.consistency_loss(logits_c_x_ulb_s,
                                               pl_ulb,
                                               'ce',
                                               mask=mask)




            if self.args.IB:
                ib_loss = self.IB_Loss(mu,logvar)
            else:
                ib_loss = torch.zeros(1).cuda(self.gpu)
            if self.args.contrastive:
                con_loss = self.contrastive_loss(torch.cat([content_proj_x_ulb_w.unsqueeze(1), content_proj_x_ulb_s.unsqueeze(1)], dim=1))
            else:
                con_loss = torch.zeros(1).cuda(self.gpu)


            # ib_loss = self.IB_Loss(mu,logvar)
            # con_loss = self.contrastive_loss(torch.cat([content_proj_x_ulb_w.unsqueeze(1), content_proj_x_ulb_s.unsqueeze(1)], dim=1))

            loss_c = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * cluster_loss + self.lambda_con * con_loss + self.lambda_ib * ib_loss


            # domain classification loss
            dom_loss = self.ce_loss(logits_s_x_ulb_w, cluster_ulb, reduction='mean')
            # if self.epoch > 0:
            if self.disentangle:
                disent_style_x_lb = self.model['model_c'].module.disentangle(style_x_lb)
                disent_style_lb_loss = -self.ce_loss(disent_style_x_lb, y_lb, reduction='mean')
                disent_style_x_ulb = self.model['model_c'].module.disentangle(style_x_ulb_w)
                disent_style_ulb_loss = -self.entropy_loss(disent_style_x_ulb)
                disent_style_loss = disent_style_lb_loss + disent_style_ulb_loss

                disent_content = self.model['model_s'].module.disentangle(content)
                disent_content_loss = -self.entropy_loss(disent_content)

            else:
                disent_style_loss = torch.zeros(1).cuda(self.gpu)
                disent_content_loss = torch.zeros(1).cuda(self.gpu)

            if self.args.MI:
                input_rec = torch.cat((x_lb, x_ulb_w))
                outputs_c_rec = self.model['model_c'](input_rec)
                content_rec, cls_mu, cls_logvar = outputs_c_rec['feat'], outputs_c_rec['mu'], outputs_c_rec['logvar']

                outputs_s_rec = self.model['model_s'](input_rec)
                style_rec, dom_mu, dom_logvar = outputs_s_rec['feat'], outputs_s_rec['mu'], outputs_s_rec['logvar']

                combined_ftr_c = torch.cat((content_rec, style_rec.detach()), dim=1)
                data_rec_c = self.model['decoder'](combined_ftr_c)
                rec_c_loss = self.rec_criterion(data_rec_c, input_rec)

                combined_ftr_s = torch.cat((content_rec.detach(), style_rec), dim=1)
                data_rec_s = self.model['decoder'](combined_ftr_s)
                rec_s_loss = self.rec_criterion(data_rec_s, input_rec)

                vae_cls = self.vae_loss(cls_mu, cls_logvar)
                vae_dom = self.vae_loss(dom_mu, dom_logvar)
                vae_cls = self.lambda_vae * vae_cls
                vae_dom = self.lambda_vae * vae_dom

                mi_c_loss = rec_c_loss + vae_cls
                mi_s_loss = rec_s_loss + vae_dom
            else:
                mi_c_loss = torch.zeros(1).cuda(self.gpu)
                mi_s_loss = torch.zeros(1).cuda(self.gpu)


            loss_c = loss_c + self.lambda_disent * disent_content_loss + mi_c_loss
            loss_s = dom_loss + self.lambda_disent * disent_style_loss + mi_s_loss

            self.update_bank(content_x_lb, content_proj_x_lb, y_lb, idx_lb)
            self.update_bank(content_x_ulb_w, content_proj_x_ulb_w, pl_ulb, idx_ulb, labeled=False)

        out_dict = self.process_out_dict(loss_c = loss_c, loss_s=loss_s)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         cluster_loss=cluster_loss.item(),
                                         ib_loss=ib_loss.item()*self.lambda_ib,
                                         con_loss=con_loss.item(),
                                         mi_c_loss=mi_c_loss.item(),
                                         disent_content_loss = disent_content_loss.item(),
                                         dom_loss=dom_loss.item(),
                                         disent_style_loss=disent_style_loss.item(),
                                         mi_s_loss=mi_s_loss.item(),
                                         loss_c = loss_c.item(),
                                         loss_s=loss_s.item(),
                                         util_ratio=mask.float().mean().item()
                                         )

        return out_dict, log_dict

    def train(self):
        """
        train function
        """
        if type(self.model) is dict:
            for name in self.model.keys():
                self.model[name].train()

        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


    def cluster_loss(self, feats_proj_x_ulb_s, centroids_sim, T=0.2):
        sim = torch.exp(torch.mm(feats_proj_x_ulb_s, self.structural_centroids.t()) / T)
        sim_probs = sim / sim.sum(1, keepdim=True)
        loss = -(torch.log(sim_probs + 1e-7) * centroids_sim).sum(1)
        loss = loss.mean()
        return loss


    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model['model_c'].load_state_dict(checkpoint['model_c'])
        self.model['model_s'].load_state_dict(checkpoint['model_s'])
        self.model['decoder'].load_state_dict(checkpoint['decoder'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']

        self.best_out_it = checkpoint['best_out_it']
        self.best_out_acc = checkpoint['best_out_acc']
        self.best_all_it = checkpoint['best_all_it']
        self.best_all_acc = checkpoint['best_all_acc']


        self.optimizer['optimizer_c'].load_state_dict(checkpoint['optimizer_c'])
        self.optimizer['optimizer_s'].load_state_dict(checkpoint['optimizer_s'])
        self.optimizer['optimizer_decoder'].load_state_dict(checkpoint['optimizer_decoder'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler['scheduler_c'].load_state_dict(checkpoint['scheduler_c'])
            self.scheduler['scheduler_s'].load_state_dict(checkpoint['scheduler_s'])
            self.scheduler['scheduler_decoder'].load_state_dict(checkpoint['scheduler_decoder'])

        # self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        self.feats_lb = checkpoint['feats_lb'].cuda(self.args.gpu)
        self.feats_proj_lb = checkpoint['feats_proj_lb'].cuda(self.args.gpu)
        self.feats_ulb = checkpoint['feats_ulb'].cuda(self.args.gpu)
        self.feats_proj_ulb = checkpoint['feats_proj_ulb'].cuda(self.args.gpu)
        self.labels_lb = checkpoint['labels_lb'].cuda(self.args.gpu)
        self.print_fn('Model loaded')

        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint



    def get_save_dict(self):
        save_dict = {
            'model_c': self.model['model_c'].state_dict(),
            'model_s': self.model['model_s'].state_dict(),
            'decoder': self.model['decoder'].state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer_c':self.optimizer['optimizer_c'].state_dict(),
            'optimizer_s': self.optimizer['optimizer_s'].state_dict(),
            'optimizer_decoder': self.optimizer['optimizer_decoder'].state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
            'best_out_it': self.best_out_it,
            'best_out_acc': self.best_out_acc,
            'best_all_it': self.best_all_it,
            'best_all_acc': self.best_all_acc,


        }
        if self.scheduler is not None:
            save_dict['scheduler_c'] = self.scheduler['scheduler_c'].state_dict()
            save_dict['scheduler_s'] = self.scheduler['scheduler_s'].state_dict()
            save_dict['scheduler_decoder'] = self.scheduler['scheduler_decoder'].state_dict()

        # save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        # save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        save_dict['feats_lb'] = self.feats_lb.cpu()
        save_dict['feats_proj_lb'] = self.feats_proj_lb.cpu()
        save_dict['feats_ulb'] = self.feats_ulb.cpu()
        save_dict['feats_proj_ulb'] = self.feats_proj_ulb.cpu()
        save_dict['labels_lb'] = self.labels_lb.cpu()

        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()

        return save_dict


    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        evaluation function
        """
        # if type(self.model) is dict:
        #     for name in self.model.keys():
        #         self.model[name].eval()

        self.model['model_c'].eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            # print(f"Evaluate on {eval_dest} with {len(eval_loader)} samples")
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model['model_c'](x)[out_key]

                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        # balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        # F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat, precision=3, suppress_small=True))
        self.ema.restore()
        self.model['model_c'].train()

        eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1}
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            # SSL_Argument('--T', float, 0.2),
            SSL_Argument('--cluster_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--use_da', str2bool, True),
            SSL_Argument('--da_len', int, 256),
            SSL_Argument('--cluster_scale', int, 100),
            SSL_Argument('--min_cluster', float, 0.9),
            SSL_Argument('--use_ema_bank', str2bool, False),
            SSL_Argument('--ema_bank_m', float, 0.7),

            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]

