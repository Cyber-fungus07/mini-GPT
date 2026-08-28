from loss import GPTLoss

class Trainer:

    def __init__(self,model,train_loader,val_loader,optimizer,device,tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer

        self.loss_fn = GPTLoss()

        self.train_losses = []
        self.val_losses = []
        self.tokens_seen = []
        self.global_step = 0

