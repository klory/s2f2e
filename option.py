import argparse
from model import create_model

class Option(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = self.add_arguments(self.parser).parse_args()

    def add_arguments(self, parser):
        parser.register("type", "bool", lambda v: v.lower() == "true")

        parser.add_argument("--img_size", type=int, default=128, help="size of input images.")
        parser.add_argument("--input_nc", type=int, default=3, help="number of channels of input(image)")
        parser.add_argument("--output_nc", type=int, default=3, help="number of channels of output(image)")
        parser.add_argument("--nfg", type=int, default=128, help="number of filters of the first conv layer of genrator.")
        parser.add_argument("--no_dropout", type="bool", nargs="?", const=True, default=False, help="whether to use dropout.")
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--out_dir", type=str, default="./out/image/", help="location to store output images.")
        parser.add_argument("--save_dir", type=str, default="./out/network/", help="location to store networks.")
        parser.add_argument("--model", type=str, default="EFG_LSGAN", help="which model to implement: EFG_WGAN, EFG_LSGAN, NFG_WGAN, NFG_LSGAN, CYC_EFG_LSGAN, CYC_EFG_WGAN")
        parser.add_argument("--lam_cyc", type=int, default=10, help="lambda to balance loss_g_gan loss_g_idt and loss_g_l1.")
        parser.add_argument("--lam_idt", type=int, default=10, help="lambda to balance loss_g_gan loss_g_idt and loss_g_l1.")
        parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam")
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
        parser.add_argument("--isTrain", type='bool', default=True, help="Specify training or testing phrase")
        parser.add_argument("--disp_freq", type=int, default=5, help="print loss freq")
        parser.add_argument("--save_freq", type=int, default=5, help="print loss freq")

        return parser

    def __call__(self):
        return self.opt

#opt = Option()()
#print type(opt)
#model = create_model(opt)
#print model
#print vars(opt)
