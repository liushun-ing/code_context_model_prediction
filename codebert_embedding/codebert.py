import gc

import math

from transformers import AutoTokenizer, AutoModel
from tokenizers import Tokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./codebert_base")
model = AutoModel.from_pretrained("./codebert_base")


# BPE_tokenizer = Tokenizer.from_file("./tokens/all_1_0.8_tokens_128.json")
# torch.Size([1, 23, 768])
# tensor([[-0.1423,  0.3766,  0.0443,  ..., -0.2513, -0.3099,  0.3183],
#         [-0.5739,  0.1333,  0.2314,  ..., -0.1240, -0.1219,  0.2033],
#         [-0.1579,  0.1335,  0.0291,  ...,  0.2340, -0.8801,  0.6216],
#         ...,
#         [-0.4042,  0.2284,  0.5241,  ..., -0.2046, -0.2419,  0.7031],
#         [-0.3894,  0.4603,  0.4797,  ..., -0.3335, -0.6049,  0.4730],
#         [-0.1433,  0.3785,  0.0450,  ..., -0.2527, -0.3121,  0.3207]],
#        grad_fn=<SelectBackward>)


def single_embedding(node: str):
    with torch.no_grad():
        model.eval()
        tokens = tokenizer.tokenize(node)
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        all_embeddings = []

        for i in range(math.floor(len(tokens_ids) / 512) + 1):
            sub_tokens = tokens_ids[i * 512: (i + 1) * 512]
            if len(sub_tokens) > 0:
                context_embeddings = model(torch.tensor(sub_tokens)[None, :])[0]
                all_embeddings.append(torch.mean(context_embeddings, dim=1))
                del context_embeddings, sub_tokens
        # if len(tokens_ids) == 0:
        #     tokens_ids = tokenizer.convert_tokens_to_ids(['[UNK]'])
        # res = torch.mean(torch.stack(all_embeddings), dim=0)
        # del all_embeddings, tokens, tokens_ids
        # print(gc.collect())
        torch.cuda.empty_cache()
        return torch.mean(torch.stack(all_embeddings), dim=0).squeeze(0)

node = """
nnn
"""
print(single_embedding(node).shape)


# def codebert(current_node: list[str]):
#     result = []
#     for node in current_node:
#         # output = BPE_tokenizer.encode(node)
#         tokens = tokenizer.tokenize(node)
#         tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
#         if len(tokens_ids) == 0:
#             tokens_ids = tokenizer.convert_tokens_to_ids(['[UNK]'])
#         context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
#         result.append(torch.mean(context_embeddings, dim=1))
#     return torch.cat(result, dim=0)
