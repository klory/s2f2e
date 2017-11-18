import model_NFG
import model_EFG

def create_model(opt):
    model = None
    if opt.model == "NFG" or opt.model == "NFG_WGAN":
        model_NFG.ModelNFG(opt)()
    elif opt.model == "EFG" or opt.model == "EFG_WGAN" or opt.model == "E2E":
        model_EFG.ModelEFG(opt)()
    else:
        raise ValueError("%s is not supported." % opt.model)
