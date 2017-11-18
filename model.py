import model_EFG
import model_NFG
import option

def create_model(opt):
    model = None
    if op.model == "EFG" or op.model == "EFG_WGAN" or op.model == "E2E":
        ModelEFG(opt)
    elif op.model == "NFG" or op.model == "NFG_WGAN":
        ModelFNG(opt)
    else:
        raise ValueError("%s is not supported." % opt.model)
