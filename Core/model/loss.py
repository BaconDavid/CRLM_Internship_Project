from torch.nn import BCELoss


class Loss:
    def __init__(self,losses=BCELoss(),*args):
        """
        args:
            args only have one loss function
        """
        self.args = args
        self.losses = losses


    def build_loss(self):
        return self.losses
    

