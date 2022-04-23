# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1


class Args:
    def __init__(self):
        self.DATA_ROOT = './data'
        self.DATA_SET_NAME = ''
        self.epoch = 100
        self.show_epoch = 1
        # self.epoch_used_for_final_result = 4
        self.model_save_epoch = 10
        self.model_save_dir = 'model_save_dir'
        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 8
        self.cuda = True
        self.opt = 'adam'
        self.lr = 1e-3  # 1e-3 5e-3
        self.weight_decay = 1e-5  # 1e-5
        self.missing_rate = 0.7
        self.missing_num = 3
        self.Vbatch_size = 128
        self.Vgpu = 0
        self.Vseed = 0
        self.Vn_hidden = 150
        self.Vdim_z = 50
        self.Vkeep_prob = 0.9
        self.Vlearning_rate = 1e-4
        self.Vepochs = 150
        self.Valpha = 1.0
        self.Vbeta = 1.0
        self.feature_dim = 2048
        self.keep_prob = 0.5
        self.scale_coeff = 1.0
        self.device = 'cpu'
        self.Vnum_class = 0
        self.smoothing_rate = 0.0


loss_coefficient = {}
