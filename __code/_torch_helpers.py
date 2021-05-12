######################################################################################
#                                       NOTES                                        #
######################################################################################

"""
Arcface algorith:
1. Normalize the embeddings and weights
2. Calculate the dot products
3. Calculate the angles with arccos
4. Add a constant factor m to the angle corresponding to the ground truth label
5. Turn angles back to cosines
6. Use cross entropy on the new cosine values to calculate loss
7. See below

Side note: there is another 7th step in the paper that scales all the vectors by a constant
factor at the end. This doesn't change the ordering of logits but changes their relative
values after softmax. I didn't see it help on this dataset but it might be a 
hyperparameter worth tweaking on other tasks.
"""

######################################################################################
#                                       CODE                                         #
######################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
import copy
import datetime
import wandb
from . import _helpers as H


class ArcFaceClassifier(nn.Module):
    "See details under notes at the top"
    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)
    
    def forward(self, x):
        # Step 1: Normalize the embeddings and weights
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2: Calculate the dot products
        return x_norm @ W_norm

def arcface_loss(cosine, targ, m=0.4):
    "See details under notes at the top"
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    # Step 3: Calculate the angles with arccos
    arcosine = cosine.arccos()
    # Step 4: Add a constant factor m to the angle corresponding to the ground truth label
    arcosine += F.one_hot(targ, num_classes = output_classes) * m
    # Step 5: Turn angles back to cosines
    cosine2 = arcosine.cos()
    # Step 6: Use cross entropy on the new cosine values to calculate loss
    return F.cross_entropy(cosine2, targ)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def seed_torch(seed=12, deterministic=True):
    torch.backends.cudnn.deterministic = deterministic # <-- If true, can drastically increase run time
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_tensor_image(image_tensor:torch.tensor, rows:int = 1, cols:int = 1, figsize=(15,10)):
    """
    param image_tensor: (samples, channels, width, height)
    """
    assert rows*cols >= 1, "rows <= 1 and cols <=1"
    assert rows*cols <= image_tensor.shape[0], f"Not enough images for {rows} rows and {cols} cols"
    assert image_tensor.shape[2] * image_tensor.shape[3] > 0, "width <= 1 and height <=1"
    if image_tensor.shape[1] > 3:
        warnings.warn("Cannot handle alpha")
    image_tensor = image_tensor.detach().cpu()


    is_grayscale = True if image_tensor.shape[1] == 1 else False
    
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    coordinates = list(itertools.product([i for i in range(rows)], [i for i in range(cols)])) # [(0, 0), (0, 1), (0, 2), (1, 0) ... ]

    for i in range(rows*cols):
        (row, col) = coordinates[i]
        
        if rows*cols == 1:
            ax = axs
        else:
            ax = axs[row, col] if (rows>1 and cols>1) else axs[i]

        image = image_tensor[i].permute(1, 2, 0).numpy()
        if np.issubdtype(image.dtype, np.float32): 
            image = np.clip(image, 0, 1)
        if is_grayscale:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.axis("off")
    plt.show()


def get_device():
    # TODO: Handle multi GPU setup ?
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_parameter_count(model, only_trainable:bool=False, decimals:int=3):
    assert decimals >= 0, "decimals most be positive"
    
    if only_trainable:
        temp = sum(p.numel() for p in model.parameters())
    else:
        temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format(temp, f".{decimals}E")


def acc_metric(pred_logits:torch.tensor, labels:torch.tensor):
    """
    pred_logits: (samples, logits)
    labels: (labels:int)
    """
    assert pred_logits.shape[0] == labels.shape[0], "number of samples in pred_logits must match that of labels"
    
    preds = torch.nn.functional.softmax(pred_logits, dim=1).argmax(dim=1)
    return (preds == labels).detach().float().mean().item()


def single_forward(model, dl):
    """
    batch[0] must be what the model expects e.g. batch = (images, labels)
    """
    sample = next(iter(dl))[0]
    model_device = next(model.parameters()).device
    sample = sample.to(model_device)
    return model(sample)


def plot_scheduler(scheduler, epochs, return_lrs=True):
    warnings.filterwarnings("ignore", category=UserWarning)
    # Ignore torch.optim user warning
    scheduler = copy.deepcopy(scheduler)
    optimizer = scheduler.optimizer
    learning_rates = []
    for epoch in range(1, epochs):
        learning_rates.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(learning_rates, '-o')
    ax.set_xticks([i for i in range(epochs)])
    ax.set_xticklabels([str(i+1) for i in range(epochs)])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_title(f"Scheduler - learning rates over {epochs} epochs")
    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Epoch")
    # reset to default warnings
    warnings.filterwarnings("default", category=UserWarning)
    return np.array([f"{lr:.3e}" for lr in learning_rates])


def get_model_save_name(to_add:dict, model_name:str, separator="  .  ", include_time=True):
    """
    Example: 
    >>> get_model_save_name({"valid_loss":valid_mean}, "model.pth")
    "time 17.25.32 03-05-2021    valid_loss 0.72153    model_name.pth"
    """
    time = datetime.datetime.now().strftime("%H.%M.%S %d-%m-%Y")
    return_string = f"time {time}{separator}" if include_time else ""
    for key, value in to_add.items():
        if type(value) in [float, np.float16, np.float32, np.float64]:
            value = f"{value:.5}".replace("+", "")

        return_string += str(key) + " " + str(value) + separator
    return_string += model_name
    assert len(return_string) < 256, "File name is to long" # Windows' char limit for file names.
    return return_string


def wandb_save_as_onnx_model(model, model_input, model_name:str = "model.onnx", path_to:str = "active_wandb"):
    """
    assert wandb is imported and active such that wandb.run.dir returns the correct path
    """
    assert H.extract_file_extension(model_name) == ".onnx", "No .onnx file extension fould"
    if path_to == "active_wandb":
        assert wandb.run is not None, "Problems with wand.run, perhaps there's no active wandb folder?"
        to_path = os.path.join(wandb.run.dir, model_name)
    else:
        path_to = ""

    torch.onnx.export(model, model_input, os.path.join(path_to, model_name))
    wandb.save(model_name)

class _Templates:
    
    def _print(self, string):
        [print(line[8:]) for line in string.split("\n")]


    def training_loop_minimal(self):
        string =\
        """
        stats = pd.DataFrame(np.zeros((C.epochs, 2)), columns=["train_loss", "valid_loss"])

        epoch_progress_bar = tqdm(range(C.epochs))
        for epoch in epoch_progress_bar:
            train_losses, valid_losses = np.zeros(len(train_dl)), np.zeros(len(valid_dl))
            
            model.train()
            for i, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
                images, labels = images.to(C.device), labels.to(C.device)
                
                # Forward pass
                preds = model(images)
                loss = C.criterion(preds, labels)
                
                # Backward pass
                loss.backward()
                
                # Batch update and logging
                optimizer.step()
                optimizer.zero_grad()
                train_losses[i] = loss.detach().cpu().item()
                if i: epoch_progress_bar.set_postfix({"train_avg":train_losses[:i].mean()})
                
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(tqdm(valid_dl, leave=False)):
                    images, labels = images.to(C.device), labels.to(C.device)
                    
                    # Forward pass
                    preds = model(images)
                    loss = C.criterion(preds, labels)
                    
                    # Batch update and logging
                    valid_losses[i] = loss.detach().cpu().item()
                    if i: epoch_progress_bar.set_postfix({"valid_avg":valid_losses[:i].mean()})
            
            # Epoch logging
            stats.iloc[epoch] = [train_losses.mean(), valid_losses.mean()]
            
        stats.plot(style="-o", figsize=(15,5))
        """

        self._print(string)

    def training_loop_with_wandb(self):
        string =\
        """
        ###################################################################################################
        #                                             Setup                                               #
        ###################################################################################################

        if C.use_wandb and not C.wandb_watch_activated: 
            C.wandb_watch_activated = True
            wandb.watch(model, C.criterion, log="all", log_freq=10)
            
        stats = pd.DataFrame(np.zeros((C.epochs,3)), columns=["train_loss", "valid_loss", "learning_rate"])
        best_model_name = "0_EPOCH.pth"
        print(H.get_gpu_memory_info())

        ###################################################################################################
        #                                          Training                                               #
        ###################################################################################################

        for epoch in tqdm(range(C.epochs)):
            train_losses, valid_losses = np.zeros(len(train_dl)), np.zeros(len(valid_dl))

            model.train()
            for i, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
                images, labels = images.to(C.device), labels.to(C.device)
                
                # Forward pass
                preds = model(images)
                loss = C.criterion(preds, labels)
                
                # Backward pass
                loss.backward()
                
                # Batch update and logging
                optimizer.step()
                optimizer.zero_grad()
                train_losses[i] = loss.detach().cpu().item()
                
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(tqdm(valid_dl, leave=False)):
                    images, labels = images.to(C.device), labels.to(C.device)
                    
                    # Forward pass
                    preds = model(images)
                    loss = C.criterion(preds, labels)
                    
                    # Batch update and logging
                    valid_losses[i] = loss.detach().cpu().item()
            
            # Epoch update and logging
            train_mean, valid_mean, lr = train_losses.mean(), valid_losses.mean(), optimizer.param_groups[0]["lr"]
            stats.iloc[epoch] = [train_mean, valid_mean, lr]
            scheduler.step()
            C.epochs_trained += 1
            
            if C.use_wandb:
                wandb.log({"train_loss": train_mean, "valid_loss": valid_mean, "lr":lr})
            if (epoch > 0) and (stats["valid_loss"][epoch] > stats["valid_loss"][epoch-1]): # Save model if it's better
                extra_info = {"valid_loss":valid_mean, "epochs_trained":C.epochs_trained}
                best_model_name = T.get_model_save_name(extra_info, "model.pth", include_time=True)
                torch.save(model.state_dict(), best_model_name)

        ###################################################################################################
        #                                          Finish up                                              #
        ###################################################################################################
                
        # Plot
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
        stats.drop(columns=["learning_rate"]).plot(ax=ax1, style="-o")
        ax2.ticklabel_format(style="scientific", axis='y', scilimits=(0,0))
        stats["learning_rate"].plot(ax=ax2, style="-o")

        # Save model
        if C.use_wandb:
            import shutil
            shutil.copy(best_model_name, os.path.join(wandb.run.dir, best_model_name))
            wandb.save(best_model_name, policy="now")
                """
        self._print(string)


    def wandb(self):
        string = \
        """
        # 1.) Config file   -->  plain and simple `dict` can be used
        # 2.) wandb.init()  -->  Starts project and spin up wandb in the background
        # 3.) wandb.watch() -->  Track the torch model's parameters and gradients over time
        # 4.) wandb.log()   -->  Logs everything else like loss, images, sounds ... (super versatile, even takes 3D objects) 
        # 5.) wandb.save()  -->  Saves the model
        """
        self._print(string)



    def config_file(self):
        string=\
        """
        class Config:
            # Control
            mode = "train"
            debug = False
            use_wandb = False
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Path
            main_path = ""; assert os.path.exists(main_path)
            csv_path = ".csv"; assert os.path.exists(csv_path)
            model_load_path = ".pth"; assert os.path.exists(model_load_path)
            
            # Adjustable variables
            wandb_watch_activated = False
            epochs_trained = 0

            # Hypers
            batch_size = 32
            epochs = 10
            criterion = nn.CrossEntropyLoss()
            optimizer_hyper = dict(lr = 5e-4)
            optimizer = torch.optim.Adam
            scheduler_hyper = dict(lr_lambda = lambda epoch: 0.85 ** epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR
            
            # Seed everything
            seed = 12
            T.seed_torch(seed)
            
            to_log = dict(
                seed = seed,
                mode = mode,
                debug = debug,
                device = device,
                epochs=epochs,
                batch_size = batch_size,
                criterion = criterion,
                optimizer = (optimizer, optimizer_hyper),
                scheduler = (scheduler, scheduler_hyper),
                dataset="MNIST",
                architecture="Resnet18",
                notes="resnet 18 is pretrained ... expect to see ... using ABC is questionable"
            )
        
        C = Config()
        H.expand_jupyter_screen(75)
        if C.use_wandb:
            wandb.login()
            wandb.init(project=?, config=C.to_log)

        """
        self._print(string)

templates = _Templates()

# Check __all__ have all function ones in a while
# [func for func, _ in inspect.getmembers(T, inspect.isfunction)]
# [func for func, _ in inspect.getmembers(T, inspect.isclass)]

__all__ =[
    # Functions
    'acc_metric',
    'arcface_loss',
    'get_device',
    'get_model_save_name',
    'get_parameter_count',
    'plot_scheduler',
    'plot_tensor_image',
    'seed_torch',
    'single_forward',
    'wandb_save_as_onnx_model',

    # Classes
    'ArcFaceClassifier',
    'RMSELoss',
    'templates'
]

