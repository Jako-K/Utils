# TODO:
"""

def get_tokenize_len_above(max_length:int, texts:iter, tokenizer, stop_at_first_violation=True):
    """
    # bert-base-uncased can take no more than 512 tokens
    # texts just need to be iterable and contrain strings
    """
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

"""


from .__code._torch_helpers import *



