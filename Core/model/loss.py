from torch.nn import CrossEntropyLoss


class Loss:
    def __init__(self,losses=CrossEntropyLoss(),*args):
        """
        args:
            args only have one loss function
        """
        self.args = args
        self.losses = losses


    def build_loss(self):
        return self.losses
    

