import dutils.type_check as _type_check
import dutils.jupyter_ipython as _jupyter_ipython
import torch
import torch as _torch
import numpy as _np
from typing import List as _List
import pandas as _pd
from datetime import datetime as _datetime
import ast as _ast


class CategoricalMetrics:
    """
    This is basically just a namespace with different functions designed to evaluate metrics e.g. accuracy.

    INPUTS:
    nc: int --> number of classes
    preds: _torch.Tensor[int64 * nc] --> contain predictions. All values must be within [0, 1, ..., nc-1]
    gt: _torch.Tensor[int64 * nc] --> contain ground truth labels. All values must be within [0, 1, ..., nc-1]
    cfm: _np.ndarray[[int32 * nc] * nc] or None --> Confusion matrix between `gt` (rows) and `preds` (columns). If `cfm=None`, the confusion matrix will be calculated automatically

    NOTE:
    I would like to have all pytorch functionality in a single file, but would also like to be able to browse different metrcis with autocomplete.
    Hence, why I made this into a class instead of ordinary functions / a seperate module.
    """

    def __init__(self):
        self.num_decimal = 5

    def _check_and_copy_input(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int, cfm: _np.ndarray = None):
        # Type checks
        _type_check.assert_types([preds, gt, nc, cfm], [_torch.Tensor, _torch.Tensor, int, _np.ndarray],
                                 [False, False, False, True])
        assert preds.dtype == _torch.int64, f"Expected predictions to be af int64 (long), but recieved `preds.dtype={preds.dtype}`"
        assert gt.dtype == _torch.int64, f"Expected ground truth labels to be af int64 (long), but recieved `preds.dtype={gt.dtype}`"

        # Value checks
        assert nc > 1, f"Expected at least 2 classes, but received `{nc}`"
        assert len(preds.shape) == 1, f"Expected predictions to be of shape 'batch_size', but received `preds.shape={preds.shape}`"
        assert len(gt.shape) == 1, f"Expected ground truth labels to be of shape 'batch_size', but received `gt.shape={gt.shape}`"
        assert gt.shape == preds.shape, "Shape mismatch between the ground truth labels and the received predictions"
        assert (gt.max() < nc) and (gt.min() >= 0), "At least one of the ground truth values are invalid"
        assert (preds.max() < nc) and (preds.min() >= 0), "At least one prediction values are is invalid"

        # Confusion matrix
        if cfm is not None:
            assert cfm.shape == (nc, nc), f"Expected the confusion matrix to be of shape `({nc, nc})`, but received `({cfm.shape})`"
            assert cfm.dtype == _np.int32, f"Expected cfm to have dtype int32, but received `cfm.dtype={cfm.dtype}`"
            cfm = cfm.copy()

        # Prepare tensors for metric calculations
        preds = preds.clone().detach().cpu().float()
        gt = gt.clone().detach().cpu().float()
        return preds, gt, cfm

    def acc(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int):
        preds, gt, _ = self._check_and_copy_input(preds, gt, nc, None)
        if nc != 2: raise NotImplementedError("Multiclass accuracy is not well defined, use recall instead.")

        acc = (preds == gt).float().mean().item()  # TODO: check implementation is correct
        return acc

    def precision(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int, cfm: _np.ndarray = None):
        # Setup
        if cfm is None: cfm = self.confusion_matrix(preds, gt, nc)
        preds, gt, cfm = self._check_and_copy_input(preds, gt, nc, cfm)

        # Precision calculation
        TP = cfm.diagonal()
        TP_plus_FP = cfm.sum(0)
        precision_per_class = TP / (TP_plus_FP + 1e-12)
        precision_per_class = precision_per_class.round(self.num_decimal)

        return {"precision_class": precision_per_class.tolist(),
                "precision_avg_micro": round(TP.sum() / TP_plus_FP.sum(), self.num_decimal),
                "precision_avg_macro": round(precision_per_class.mean(), self.num_decimal)}

    def recall(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int, cfm: _np.ndarray = None):
        # Setup
        if cfm is None: cfm = self.confusion_matrix(preds, gt, nc)
        preds, gt, cfm = self._check_and_copy_input(preds, gt, nc, cfm)

        # Recall calculation
        TP = cfm.diagonal()
        TP_plus_FN = cfm.sum(1)
        recall_per_class = TP / (TP_plus_FN + 1e-12)
        recall_per_class = recall_per_class.round(self.num_decimal)

        return {"recall_class": recall_per_class.tolist(),
                "recall_avg_micro": round(TP.sum() / TP_plus_FN.sum(), self.num_decimal),
                "recall_avg_macro": round(recall_per_class.mean(), self.num_decimal)}

    def f1_score(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int, cfm: _np.ndarray = None):
        # Setup
        if cfm is None: cfm = self.confusion_matrix(preds, gt, nc)
        preds, gt, cfm = self._check_and_copy_input(preds, gt, nc, cfm)

        # F1 score calculation (macro)
        precision = _np.array(self.precision(preds.long(), gt.long(), nc, cfm)["precision_class"])
        recall = _np.array(self.recall(preds.long(), gt.long(), nc, cfm)["recall_class"])
        f1_per_class = (2 * precision * recall) / (precision + recall + 1e-12)
        f1_per_class = f1_per_class.round(self.num_decimal)

        # F1 score calculation (micro)
        # So to make a long story short "f1=precision=recall" is true in multiclass setups with micro averging.
        # The reason for this is essentailly that FP=FN ==> precision = recall.
        # This was kinda wierd to me at first, but in a multiclass setup all the elements in a conf. matrix is both FP and FN simulationously i.e. a wrong prediction in one class will always be missing in another
        # So I have just calculated the precision ones to avoid unnecessary computations
        TP = cfm.diagonal().sum()
        FP = cfm.sum(0).sum() - TP  # == cfm.sum(1).sum() - TP
        f1_avg_micro = round((TP / (TP + FP)).mean(), self.num_decimal)

        return {"f1_class": f1_per_class.tolist(),
                "f1_avg_micro": round(f1_avg_micro, self.num_decimal),
                "f1_avg_macro": round(f1_per_class.mean(), self.num_decimal)}

    def class_balance(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int, label_names: _List[str],
                      plot_class_dist: bool = False):
        # _np.unique(labels.numpy(), return_counts=True)
        # plot_class_dist
        # translate numbers to names
        raise NotImplementedError("")

    def confusion_matrix(self, preds: _torch.Tensor, gt: _torch.Tensor, nc: int):
        preds, gt, _ = self._check_and_copy_input(preds, gt, nc, None)
        cfm = _np.zeros((nc, nc))
        for p, l in zip(preds.long(), gt.long()):
            cfm[l, p] += 1
        return cfm.astype(int)


pytorch_metrics = CategoricalMetrics()


class CategoricalLogger:
    """


    # EXAMPLE (1)
    >> logger = CategoricalLogger(1000, 3)
    >> for epoch in range(5):
    >>    logger.update(epoch, _torch.randint(0, 3, (10000,)).long(), _torch.randint(0, 3, (10000,)).long())
    >> print(logger)
    >> logger.get_overall_average()
    
    
    # EXAMPLE (2)
    >> logger = CategoricalLogger(10, 3)
    >> for epoch in range(5):
    >>     logger.update(epoch, _torch.randint(0, 3, (100,)).long(), _torch.randint(0, 3, (100,)).long(),
    >>                   _torch.rand(1)[0], _torch.rand(1)[0])
    >> print(logger)
    >> logger.get_overall_average()

    >> predictions = _torch.tensor([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]).long()
    >> labels = _torch.tensor([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]).long()

    >> print(" ", predictions, "\n ", labels)
    >> C = CategoricalMetrics()
    >> print(C.confusion_matrix(predictions, labels, 3))
    >> print(C.precision(predictions, labels, 3))
    >> print(C.recall(predictions, labels, 3))
    >> print(C.f1_score(predictions, labels, 3))
    """




    def __init__(self, batch_size: int, num_classes: int, epochs_trained_prior: int = 0, acc: bool = False,
                 precision: bool = True, recall: bool = True, f1: bool = True, confusion_matrix: bool = True):
        if acc and not (num_classes != 2):
            raise ValueError("Accuracy is only defined for binary classification tasks")
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs_trained_prior = epochs_trained_prior

        # Prepare metrics
        extra_cols = []
        self.metrics = []
        if acc:
            extra_cols += ["acc"]
            self.metrics.append("acc")
        if precision:
            extra_cols += ['precision_class', 'precision_avg_micro', 'precision_avg_macro']
            self.metrics.append("precision")
        if recall:
            extra_cols += ['recall_class', 'recall_avg_micro', 'recall_avg_macro']
            self.metrics.append("recall")
        if f1:
            extra_cols += ['f1_class', 'f1_avg_micro', 'f1_avg_macro']
            self.metrics.append("f1")
        if confusion_matrix:
            extra_cols += ["confusion_matrix"]
            self.metrics.append("confusion_matrix")

        df_columns = ["timestamp", "epoch_trained_relative", "epochs_trained_total", "loss_train", "loss_valid"]
        self.df = _pd.DataFrame(columns=df_columns)


    def __repr__(self):
        if _jupyter_ipython.in_jupyter():
            display(self.df)  # This will just display the pandas dataframe as normally in jupyter notebook
            return ""
        else:
            return str(self.df)

    def _calculate_metric(self, metric, preds, gt):
        cfm = pytorch_metrics.confusion_matrix(preds, gt, self.num_classes)
        if metric == "acc":
            return pytorch_metrics.acc(preds, gt, self.num_classes)
        if metric == "precision":
            return pytorch_metrics.precision(preds, gt, self.num_classes, cfm=cfm)
        if metric == "recall":
            return pytorch_metrics.recall(preds, gt, self.num_classes, cfm=cfm)
        if metric == "f1":
            return pytorch_metrics.f1_score(preds, gt, self.num_classes, cfm=cfm)
        if metric == "confusion_matrix":
            return cfm

    def update(self, current_epoch, preds: _torch.Tensor, gt: _torch.Tensor, loss_train:_torch.Tensor=None, loss_valid:_torch.Tensor=None):
        # Checks
        _type_check.assert_types([current_epoch, preds, gt, loss_train, loss_valid], [int] + [_torch.Tensor]*4, [False, False, False, True, True])

        # Add new row which will be populated one variable at a time
        i = len(self.df)
        self.df.loc[i] = None

        # Simple logging stuff
        self.df.loc[i, "epoch_trained_relative"] = current_epoch
        self.df.loc[i, "epochs_trained_total"] = self.epochs_trained_prior + current_epoch
        self.df.loc[i, "timestamp"] = _datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        # Losses
        if loss_train:
            self.df.loc[i, "loss_train"] = loss_train.clone().cpu().detach().item()
            assert loss_valid.shape == _torch.Size([]), f"Expected `loss_valid` be a single number, but received `loss_train.shape={loss_valid.shape}`"
        if loss_valid:
            self.df.loc[i, "loss_valid"] = loss_valid.clone().cpu().detach().item()
            assert loss_train.shape == _torch.Size([]), f"Expected `train_loss` be a single number, but received `loss_train.shape={loss_train.shape}`"

        # Calculate all the metrics and add them one at a time.
        # Some metrics return more then one value (per class, avg_micro, ...). This is handled with dicts
        for metric in self.metrics:
            return_value = self._calculate_metric(metric, preds, gt)
            if isinstance(return_value, dict):
                for name, value in return_value.items():
                    self.df.loc[i, name] = str(value)
            elif metric == "confusion_matrix":
                self.df.loc[i, metric] = str(return_value.tolist())
            else:
                self.df.loc[i, metric] = str(return_value)

        # Check if there's any illegal NAs (only train_loss and valid_loss is allowed to be NA, hence the drop)
        assert not any(self.df.drop(columns=["loss_train", "loss_valid"]).iloc[i].isna().tolist()), \
            f"At least one value was determined to be NA. The problem occurred in row: {self.df.loc[i]}"

    def get_overall_average(self):
        df_combined = self.df.iloc[0:0].copy()
        df_combined = df_combined.drop(columns=["timestamp"])
        df_combined.loc[0] = None

        for col_name in df_combined:
            values_combined = _np.array([_ast.literal_eval(str(l)) for l in self.df[col_name].tolist() if str(l) != "nan"])
            if len(values_combined) == 0:
                if col_name not in ["loss_train", "loss_valid"]: raise RuntimeError("Unexpected error. Probably caused by illegal NAs")
                df_combined[col_name] = None
            elif col_name in ["epoch_trained_relative", "epochs_trained_total"]:
                df_combined[col_name] = self.df[col_name].max()
            elif "class" in col_name:
                df_combined[col_name] = str(values_combined.mean(0))
            elif col_name == "confusion_matrix":
                df_combined[col_name] = str(values_combined.sum(0).tolist())
            else:
                df_combined[col_name] = values_combined.mean()
        return df_combined


if __name__ == "__main__" and False:
    logger = CategoricalLogger(10, 3)
    for epoch in range(5):
        logger.update(epoch, _torch.randint(0, 3, (100,)).long(), _torch.randint(0, 3, (100,)).long(),
                      _torch.rand(1)[0], _torch.rand(1)[0])
    print(logger)
    logger.get_overall_average()


    predictions = _torch.tensor([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]).long()
    labels = _torch.tensor([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]).long()

    print(" ", predictions, "\n ", labels)
    C = CategoricalMetrics()
    print(C.confusion_matrix(predictions, labels, 3))
    print(C.precision(predictions, labels, 3))
    print(C.recall(predictions, labels, 3))
    print(C.f1_score(predictions, labels, 3))