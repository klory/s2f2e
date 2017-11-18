import argparse
from model import create_model

class Option(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = self.add_arguments(self.parser).parse_args()

    def add_arguments(self, parser):
        parser.register("type", "bool", lambda v: v.lower() == "true")

        parser.add_argument("--input_nc", type=int, default=3, help="number of channels of input(image)")
        parser.add_argument("--output_nc", type=int, default=3, help="number of channels of output(image)")
        parser.add_argument("--nfg", type=int, default=128, help="number of filters of the first conv layer of genrator.")
        parser.add_argument("--no_dropout", type="bool", nargs="?", const=True, default=False, help="whether to use dropout.")
        parser.add_argument("--use_sigmoid", type="bool", nargs="?", const=True, default=False, help="whether to use sigmoid.")
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--optimizer", type=str, default="adam", help="adam or RMSProp")
        parser.add_argument("--out_dir", type=str, default="./out/image/", help="location to store output images.")
        parser.add_argument("--model", type=str, default="E2E", help="which model to implement: EFG, NFG, NFG_WGAN, EFG_WGAN, WGAN, E2E")
        parser.add_argument("--lam", type=int, default=10, help="lambda to balance loss_g_gan and loss_g_l1.")
        parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam")
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")

        return parser

    def __call__(self):
        return self.opt

opt = Option()()
#print type(opt)
create_model(opt)
#print vars(opt)
