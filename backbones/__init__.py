from .iresnet import iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    # resnet
    if name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()
