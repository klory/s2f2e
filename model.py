import model_cycle
import model_Unet

def create_model(opt):
    model = None
    if 'CYC' in opt.model:
        model = model_cycle.Cycle()
    elif opt.model == 'EFG_WGAN' or opt.model == 'EFG_LSGAN' or opt.model == 'NFG_WGAN' or opt.model == 'NFG_LSGAN':
        model = model_Unet.Unet()
    else:
        raise ValueError("%s is not supported." % opt.model)
    model.initialize(opt)
    return model
