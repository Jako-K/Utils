# TODO:
"""

# This should probably just be get longest token instead, right?
def get_tokenize_len_above(max_length:int, texts:iter, tokenizer, stop_at_first_violation=True):
    #
    # bert-base-uncased can take no more than 512 tokens
    # texts just need to be iterable and contrain strings
    #
    return_dict = {}
    for i, text in enumerate(tqdm(texts)):
        length = len(tokenizer.tokenize(train_df.excerpt[0]))
        if length > 512:
            return_dict[i] = length
            if not stop_at_first_violation:return return_dict
    return return_dict


ADD TO TEMPLATES:
- dataset and dataloader in the same template

CONFIG
- workers
- train_valid split (also in to_wandb)


# Add BERT-utils somehow making it easier to tokenize --> model

# Add local logging (tqdm) to wandb train_loop. This is needed because it only looks after a full epoch, which is way to late to detect any bugs.
# Add some sort of timer logging i wandb training_loop

# Examine if wandb.watch(model, C.criterion, log="all", log_freq=len(train_dl)+len(valid_dl)) is signifcantly faster, if so add that instead of current every 10 batches update.

# Templates --> Fix save best model in such that it dosen't look at the previous one but at the all time best
# - Check saved epoch number as well, think it's off by one

# ADD pin_memory = True in dataloaders

# ADD wandb.run.summary["best_valid_loss", "best_valid_loss_epoch"] to training loop and in general update it

# ADD multiple saved model in training loop such that perhaps the 3 best models get saved as opposed to only the very best

"""

from .__code._torch_helpers import *
