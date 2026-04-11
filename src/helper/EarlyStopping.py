import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, path='checkpoint.pth', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None  # NEW: Store weights in memory

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.update_best_weights(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.update_best_weights(model)
            self.counter = 0

    def update_best_weights(self, model):
        self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if self.verbose:
            print(f"New best loss found. Best weights updated in memory.")

    def save_checkpoint(self):
        if self.best_model_weights is not None:
            if self.verbose:
                print(f"Saving FINAL model checkpoint to {self.path}")
            torch.save(self.best_model_weights, self.path)