def resnext101_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['resnext101_32x4d'](pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])
    

_resnext_meta = {'cut': -2, 'split': lambda m: (m[0][6], m[1]) }
model_meta[resnext101_32x4d] = _resnext_meta

def se_resnet50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['se_resnet50'](pretrained=pretrained)
    return model
    
_se_resnet_meta = {'cut':-2, 'split': lambda m: (m[0][3], m[1])}
model_meta[se_resnet50] = _se_resnet_meta
