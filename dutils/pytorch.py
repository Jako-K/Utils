
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import random as _random
import os as _os
import cv2 as _cv2
import warnings as _warnings
import itertools as _itertools
import datetime as _datetime
import numpy as _np
import matplotlib.pyplot as _plt
import wandb as _wandb

from . import type_check as _type_check
from . import system_info as _system_info
from . import colors as _colors



class ArcFaceClassifier(_nn.Module):
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

    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = _nn.Parameter(_torch.Tensor(emb_size, output_classes))
        _nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # Step 1: Normalize the embeddings and weights
        x_norm = _F.normalize(x)
        W_norm = _F.normalize(self.W, dim=0)
        # Step 2: Calculate the dot products
        return x_norm @ W_norm

#TODO add checks
def arcface_loss(cosine, target, output_classes, m=0.4):
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

    # this prevents NaN when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
    # Step 3: Calculate the angles with arccos
    arcosine = cosine.arccos()
    # Step 4: Add a constant factor m to the angle corresponding to the ground truth label
    arcosine += _F.one_hot(target, num_classes=output_classes) * m
    # Step 5: Turn angles back to cosines
    cosine2 = arcosine.cos()
    # Step 6: Use cross entropy on the new cosine values to calculate loss
    return _F.cross_entropy(cosine2, target)

#TODO add checks
class RMSELoss(_nn.Module):
    """ Root mean square error loss """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = _nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = _torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class _Templates:

    @staticmethod
    def _print(string):
        [print(line[12:]) for line in string.split("\n")]  # Remove the first 8 spaces from print


    def scheduler_plotter(self):
        string = \
            """
            # !git clone https://www.github.com/Jako-K/schedulerplotter
            from schedulerplotter import Plotter
            Plotter();
            """
        self._print(string)


    def common_album_aug(self):
        string = \
            """
            from albumentations.pytorch import ToTensorV2
            from albumentations import (
                HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
                IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
                IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, 
                ShiftScaleRotate, CenterCrop, Resize, MultiplicativeNoise, Solarize, MotionBlur
            )
    
            augmentations = Compose([
                RandomResizedCrop(height?, width?),
                ToTensorV2()
            ])
    
            aug_image = augmentations(image=IMAGE_NAME?)['image']
            """
        self._print(string)


    def training_loop_minimal(self):
        string = \
            """
            stats = pd.DataFrame(np.zeros((C.epochs, 2)), columns=["train_loss", "valid_loss"])
    
            epoch_progress_bar = tqdm(range(C.epochs))
            for epoch in epoch_progress_bar:
                train_losses, valid_losses = np.zeros(len(dl_train)), np.zeros(len(dl_valid))
    
                model.train()
                for i, (images, labels) in enumerate(tqdm(dl_train, leave=False)):
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
                    for i, (images, labels) in enumerate(tqdm(dl_valid, leave=False)):
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
        string = \
            """
            ###################################################################################################
            #                                             Setup                                               #
            ###################################################################################################
    
            if C.use_wandb and not C.wandb_watch_activated: 
                C.wandb_watch_activated = True
                wandb.watch(model, C.criterion, log="all", log_freq=10)
    
            stats = pd.DataFrame(np.zeros((C.epochs,3)), columns=["train_loss", "valid_loss", "learning_rate"])
            best_model_name = "0_EPOCH.pth"
            print(U.system_info.get_gpu_info())
    
            ###################################################################################################
            #                                          Training                                               #
            ###################################################################################################
    
            for epoch in tqdm(range(C.epochs)):
                train_losses, valid_losses = np.zeros(len(dl_train)), np.zeros(len(dl_valid))
    
                model.train()
                for i, (images, labels) in enumerate(tqdm(dl_train, leave=False)):
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
                    for i, (images, labels) in enumerate(tqdm(dl_valid, leave=False)):
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
    
                if (epoch > 0) and (stats["valid_loss"][epoch] < stats["valid_loss"][epoch-1]): # Save model if it's better
                    if C.save_only_best_model and (best_model_name != "0_EPOCH.pth"): 
                        U.input_output.remove_file(f"./{best_model_name}")
                        
                    extra_info = {"valid_loss":valid_mean, "epochs_trained":C.epochs_trained}
                    best_model_name = U.pytorch.get_model_save_name("model.pth", extra_info, include_time=True)
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


    def config_file(self, on_kaggle=False):
        login = \
            """
                from kaggle_secrets import UserSecretsClient
                wandb.login(key=UserSecretsClient().get_secret("wandb_api_key"))
            """ if on_kaggle else _wandb.login()
        string = \
            f"""
            class Config:
                # Control
                mode = "train"
                debug = True
                use_wandb = False
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                validation_percentage = 0.10
                save_only_best_model = True
                
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
                architecture = "efficientnet_b3"
                
                # Seed everything
                U.pytorch.seed_torch(seed = 12, deterministic = False)
                
                # W&B logging
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
                    notes="efficientnet_b3 is pretrained ... expect to see ... using ABC is questionable"
                )
    
            C = Config()
            if C.use_wandb:
                {login}
                wandb.init(project=?, config=C.to_log)
    
            """
        self._print(string)
templates = _Templates()

def seed_torch(seed:int, deterministic:bool = False):
    """
    Function description:
    Seed python, random, os, bumpy, torch and torch.cuda.

    @param seed: Used to seed everything
    @param deterministic: Set `torch.backends.cudnn.deterministic`. NOTE can drastically increase run time if True


    """
    _type_check.assert_types([seed, deterministic], [int, bool])

    _torch.backends.cudnn.deterministic = deterministic
    _random.seed(seed)
    _os.environ['PYTHONHASHSEED'] = str(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)


def show_tensor_image(image_tensor: _torch.Tensor, rows: int = 1, cols: int = 1, fig_size:tuple = (15, 10)):
    """
    Function description:
    Can show multiple tensor images.

    @param image_tensor: (samples, channels, width, height)
    @param rows: Number of rows in the plot
    @param cols: Number of columns in the plot
    @param fig_size: matplotlib.pyplot figure size (width x height)
    """
    # Checks
    _type_check.assert_types([image_tensor, rows, cols, fig_size],
                             [_torch.Tensor, int, int, tuple])
    if (rows < 1) or (cols < 1):
        raise ValueError("Both `rows` and `cols` must be greater than or equal to 1")
    if rows * cols > image_tensor.shape[0]:
        raise ValueError(f"Not enough images for {rows} rows and {cols} cols")
    if len(image_tensor.shape) != 4:
        raise ValueError(f"Expected shape (samples, channels, width, height) received {image_tensor.shape}. "
                         f"If greyscale try `image_tensor.unsqueeze(1)`")
    if image_tensor.shape[1] > 3:
        _warnings.warn("Cannot handle alpha channels, they will be ignored")

    # Just to be sure
    image_tensor = image_tensor.detach().cpu()

    # If any_pixel_value > 1 --> assuming image_tensor is in [0,255] format and will normalize to [0,1]
    if sum(image_tensor > 1.0).sum() > 0:
        image_tensor = image_tensor / 255

    # Prepare for loop
    is_grayscale = True if image_tensor.shape[1] == 1 else False
    _, axs = _plt.subplots(rows, cols, figsize=fig_size)
    # coordinates = [(0, 0), (0, 1), (0, 2), (1, 0) ... ]
    coordinates = list(_itertools.product([i for i in range(rows)], [i for i in range(cols)]))

    for i in range(rows * cols):
        (row, col) = coordinates[i]

        # Deal with 1D or 2D plot i.e. multiple columns and/or rows
        if rows * cols == 1:
            ax = axs
        else:
            ax = axs[row, col] if (rows > 1 and cols > 1) else axs[i]

        # Format shenanigans
        image = image_tensor[i].permute(1, 2, 0).numpy()
        if _np.issubdtype(image.dtype, _np.float32):
            image = _np.clip(image, 0, 1)

        # Actual plots
        if is_grayscale:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.axis("off")
    _plt.show()


def get_device():
    # TODO: Handle multi GPU setup ?
    return _torch.device("cuda:0" if _torch.cuda.is_available() else "cpu")


def get_parameter_count(model: _nn.Module, only_trainable: bool = False, decimals: int = 3):
    """ Number of total or trainable parameters in a pytorch model i.e. a nn.Module child """
    _type_check.assert_types([model, only_trainable, decimals], [_nn.Module, bool, int])
    if decimals < 1:
        raise ValueError(f"Expected `decimals` >= 1, but received `{decimals}`")

    if only_trainable:
        temp = sum(p.numel() for p in model.parameters())
    else:
        temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format(temp, f".{decimals}E")


def metric_acc(pred_logits: _torch.Tensor, targets: _torch.Tensor):
    """
    Function description:
    Calculate accuracy of logits

    @param pred_logits: Prediction logits used to calculate the accuracy. Expected shape: (samples, logits)
    @param targets: Ground truth targets (ints)
    """
    _type_check.assert_types([pred_logits, targets], [_torch.Tensor, _torch.Tensor])
    if pred_logits.shape[0] != targets.shape[0]:
        raise ValueError("Dimension mismatch between `pred_logits` and `targets`")

    preds = _torch._nn.functional.softmax(pred_logits, dim=1).argmax(dim=1)
    return (preds == targets).detach().float().mean().item()


def get_batch(dataloader: _torch.utils.data.DataLoader):
    """ Get the next batch """
    _type_check.assert_type(dataloader, _torch.utils.data.DataLoader)
    return next(iter(dataloader))


def get_model_save_name(model_name:str, to_add: dict=None, separator: str = "  .  ", include_time: bool = True):

    """
    Function description:
    Adds useful information to model save file such as date, time and metrics.

    Example:
    >> get_model_save_name("model.pth", {"valid_loss":123}, "  |  ")
    "time 17.25.32 03-05-2021  |  valid_loss 0.72153  |  model_name.pth"

    @param to_add: Dictionary which contain information which will be added to the model save name e.g. loss
    @param model_name: Actual name of the model. Will be the last thing appended to the save path
    @param separator: The separator symbol used between information e.g. "thing1 <separator> thing2 <separator> ...
    @param include_time: If true, include full date and time  e.g. 17.25.32 03-05-2021 <separator> ...
    """
    # Checks
    if to_add is None: to_add = {} # Avoid mutable default arguments
    _type_check.assert_types([model_name, to_add, separator, include_time], [str, dict, str, bool])
    if _system_info.on_windows() and (separator in _system_info.windows_illegal_file_name_character):
        raise ValueError(f"Received illegal separator symbol `{separator}`. Windows don't allow the "
                         f"following characters in a filename: {_system_info.windows_illegal_file_name_character}")

    return_string = ""
    if include_time:
        time_plus_date = _datetime.datetime.now().strftime('%H.%M.%S %d-%m-%Y')
        return_string = f"time {time_plus_date}{separator}" if include_time else ""

    # Adds everything from to_add dict
    for key, value in to_add.items():
        if type(value) in [float, _np.float16, _np.float32, _np.float64]:
            value = f"{value:.5}".replace("+", "")  # Rounding to 5 decimals
        return_string += str(key) + " " + str(value) + separator

    return_string += model_name
    if _system_info.on_windows():
        if len(return_string) > 256:
            raise RuntimeError(f"File name is to long. Windows allows non more than 256 character, "
                               f"but attempted to save file with `{len(return_string)}` characters")

    return return_string


def yolo_bb_from_normal_bb(bb, img_width, img_height, label, xywh=False):
    # TODO refactor and add checks
    if not xywh:
        x1, y1, x2, y2 = bb
        bb_width, bb_height = (x2 - x1), (y2 - y1)
    else:
        x1, y1, bb_width, bb_height = bb

    # Width and height
    bb_width_norm = bb_width / img_width
    bb_height_norm = bb_height / img_height

    # Center
    bb_center_x_norm = (x1 + bb_width / 2) / img_width
    bb_center_y_norm = (y1 + bb_height / 2) / img_height

    # Yolo format --> |class_name center_x center_y width height|.txt  -  NOT included the two '|'
    string = str(label)
    for s in [bb_center_x_norm, bb_center_y_norm, bb_width_norm, bb_height_norm]:
        string += " " + str(s)

    return string


def yolo_draw_bbs_path(yolo_image_path, yolo_bb_path, color=(0, 0, 255)):
    # TODO refactor and add checks
    assert _os.path.exists(yolo_image_path), "Bad path"
    image = _cv2.imread(yolo_image_path)
    dh, dw, _ = image.shape

    fl = open(yolo_bb_path, "r")
    data = fl.readlines()
    fl.close()

    for dt in data:
        _, x, y, w, h = map(float, dt.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0: l = 0
        if r > dw - 1: r = dw - 1
        if t < 0: t = 0
        if b > dh - 1: b = dh - 1

        _cv2.rectangle(image, (l, t), (r, b), color, 2)
    return image


def yolo_draw_single_bb_cv2(image_cv2, x, y, w, h, color=(0, 0, 255)):
    # TODO refactor and add checks. Also merge this with `yolo_draw_bbs_path` seems wasteful to have both
    dh, dw, _ = image_cv2.shape

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0: l = 0
    if r > dw - 1: r = dw - 1
    if t < 0: t = 0
    if b > dh - 1: b = dh - 1

    _cv2.rectangle(image_cv2, (l, t), (r, b), color, 2)
    return image_cv2


def fold_performance_plot(data:_np.ndarray, stds: int = 1):
    """
    Illustrates the performance of each fold individually and as a whole.

    EXAMPLE:
    # 3 folds each trained for 4 epochs
    >> data = np.array([[1,2,3,4],
                        [2,1,4,5],
                        [3,2,5,1]])
    >> fold_performance_plot(data)

    @param data: 2D array where the rows are performance data for each fold and the columns are corresponding epochs
    @param stds: How many standard deviations the combined fold plot is going to have
    @return: matplotlib.pyplot.fig
    """

    xs = _np.arange(data.shape[1])
    std = data.std(0)
    mean = data.mean(0)

    fig, (ax1, ax2) = _plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax1.set_title("Individually")
    for i, d in enumerate(data):
        ax1.plot(d, ".-", label=f"fold {i}")
    ax1.legend()

    blue, orange = _colors.get_colors(["blue", "orange"], color_type="rgb_01")
    ax2.set_title("Averaged with uncertainty")
    ax2.plot(mean, 'o-', color=blue, label='Mean')
    fig.gca().fill_between(xs, mean - stds * std, mean + stds * std, color=blue,
                            alpha=0.2, label=str(stds) + r"$\cdot\sigma$")
    ax2.plot(xs, [mean.mean()] * len(xs), '--', color=orange, label="Mean of means")
    ax2.set_xticks(xs)
    ax2.set_xlabel("Epochs")
    ax2.legend()

    _plt.show()
    return fig


from torch.utils.data import DataLoader as _DataLoader
class DataLoaderPlus:
    """
    A simple wrapper class for pytorch's DataLoader class.
    The intended use is a to provide a somewhat simple `preprocess_func`
    e.g. .to_device()
    """
    def __init__(self, dl:_DataLoader, preprocess_func:_type_check.FunctionType):
        self.dl = dl
        self.f = preprocess_func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.f(*b)



__all__ = [
    "ArcFaceClassifier",
    "arcface_loss",
    "RMSELoss",
    "templates",
    "seed_torch",
    "show_tensor_image",
    "get_device",
    "get_parameter_count",
    "metric_acc",
    "get_batch",
    "get_model_save_name",
    "yolo_bb_from_normal_bb",
    "yolo_draw_bbs_path",
    "yolo_draw_single_bb_cv2",
    "fold_performance_plot",
]
