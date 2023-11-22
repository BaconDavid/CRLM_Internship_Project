from torch.nn.functional import cross_entropy

def build_loss():
    return cross_entropy()