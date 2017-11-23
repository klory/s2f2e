import argparse
from model import create_model

class Option(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = self.add_arguments(self.parser).parse_args()

    def add_arguments(self, parser):
        parser.register("type", "bool", lambda v: v.lower() == "true")

        parser.add_argument("--img_size", type=int, default=128, help="size of input images.")
        parser.add_argument("--epoch_num", type=int, default=100, help="number of epochs")
        parser.add_argument("--input_nc", type=int, default=3, help="number of channels of input(image)")
        parser.add_argument("--output_nc", type=int, default=3, help="number of channels of output(image)")
        parser.add_argument("--nfg", type=int, default=64, help="number of filters of the first conv layer of genrator.")
        parser.add_argument("--no_dropout", type="bool", nargs="?", const=True, default=False, help="whether to use dropout.")
        parser.add_argument("--batch_size", type=int, default=8, help="batch size")
        parser.add_argument("--out_dir", type=str, default="./out/image/", help="location to store output images.")
        parser.add_argument("--save_dir", type=str, default="./out/network/", help="location to store networks.")
        parser.add_argument("--model", type=str, default="EFG_WGAN", help="which model to implement: EFG_WGAN, EFG_LSGAN, NFG_WGAN, NFG_LSGAN, CYC_EFG_LSGAN, CYC_EFG_WGAN")
        parser.add_argument("--lam_cyc", type=int, default=10, help="lambda to balance loss_g_gan loss_g_idt and loss_g_l1.")
        parser.add_argument("--lam_idt", type=int, default=10, help="lambda to balance loss_g_gan loss_g_idt and loss_g_l1.")
        parser.add_argument("--lam_l1", type=int, default=10, help="lambda to balance loss_g_gan loss_g_idt and loss_g_l1.")
        parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam")
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
        parser.add_argument("--isTrain", type='bool', default=True, help="Specify training or testing phrase")
        parser.add_argument("--disp_freq", type=int, default=5, help="print loss freq, unit: iteration")
        parser.add_argument("--save_freq", type=int, default=100, help="print loss freq, unit: epoch")
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--is_small', type='bool', default=False, help='whether to choose small set to test overfit')


        return parser

    def __call__(self):
        return self.opt

#opt = Option()()
#print type(opt)
#model = create_model(opt)
#print model
#print vars(opt)
