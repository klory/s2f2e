import torch
import os
import numpy as np

class BaseModel():
    def name(self, opt):
        return opt.model
    
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = opt.save_dir
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

    def set_input(self):

        pass

    def forward(self):
        pass

    def test(self):
        pass

    def optimizer(self):
        pass

    def get_loss(self):
        pass

    def label_generate(self, label, batch_size):
        v1 = np.array([1] * 8 * 8).reshape(1, 1, 8, 8)
        v0 = np.array([0] * 8 * 8).reshape(1, 1, 8, 8)
        if label == 0:
            v = np.concatenate((v1, v0, v0), axis=1)
            v = np.repeat(v, batch_size, axis=0)
        elif label == 1:
            v = np.concatenate((v0, v1, v0), axis=1)
            v = np.repeat(v, batch_size, axis=0)
        if label == 2:
            v = np.concatenate((v0, v0, v1), axis=1)
            v = np.repeat(v, batch_size, axis=0)

        return self.Tensor(v.astype(np.float))
    
    def save_network(self, network, network_label, epoch_label):
        save_file = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_file)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            # TODO: ??? what is doing here
            network.cuda()

    def load_network(self, network, network_label, epoch_label):
        save_file = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_file)
        network.load_state_dict(torch.load(save_path))

