import torch
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
# tokenizer._tokenizer.post_processor = TemplateProcessing(
#     single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
#     pair=tokenizer.bos_token+" $A "+tokenizer.eos_token+" "+tokenizer.bos_token +" $B " +tokenizer.eos_token,
#     special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
# )


def tokenize_inputs(clones1, clones2):
    tokenized_inputs = tokenizer(
        [clones1, clones2],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=400,
    ).to(device)
    # words1 = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])
    tokenized_inputs.input_ids = torch.flatten(tokenized_inputs.input_ids)
    tokenized_inputs.attention_mask = torch.flatten(tokenized_inputs.attention_mask)
    return tokenized_inputs


def tokenization(row):
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=400,
    )
    tokenized_inputs.input_ids = torch.flatten(tokenized_inputs.input_ids)
    tokenized_inputs.attention_mask = torch.flatten(tokenized_inputs.attention_mask)
    return tokenized_inputs


# ROBERTA: 257 LIMIT
def tokenization(row):
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=257,
    )
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    #     print(tokenized_inputs.shape)
    return tokenized_inputs


def tokenization(row):
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=257,
    )
    # Reshape the tensor from [2, 257] to [1, 514]
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].view(1, -1)
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].view(1, -1)
    return tokenized_inputs


#### Final one:
def tokenization(row):
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=255,
    )
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    #     print(tokenized_inputs.shape)
    return tokenized_inputs
