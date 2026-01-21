
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.best_auc = 0
        self.epochs_without_improvement = 0
        self.stop_training = False

    def __call__(self, val_loss,val_auc):
        # if val_loss < self.best_loss - self.delta:
        #     self.best_loss = val_loss
        #     self.epochs_without_improvement = 0
        # else:
        #     self.epochs_without_improvement += 1

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.stop_training = True