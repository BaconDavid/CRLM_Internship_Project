from torch.nn import CrossEntropyLoss


class Loss:
    def __init__(self,losses,*args):
        """
        args:
            args only have one loss function
        """
        self.args = args
        self.losses = losses


    def build_loss(self):
        return self.args[0]
    
    def calculate_loss(self,*args):
        self.losses += self.args[0].item()