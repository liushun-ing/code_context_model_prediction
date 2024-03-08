# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#
# from tokenizers.trainers import BpeTrainer
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# data = ["low", "lower", "newest", "wider"]
# from tokenizers.pre_tokenizers import Whitespace
# tokenizer.pre_tokenizer = Whitespace()
#
# # files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
# # tokenizer.train(files, trainer)
# tokenizer.train_from_iterator(data, trainer)
#
# tokenizer.save("tokenizer-wiki.json")
#
# tokenizer = Tokenizer.from_file("tokenizer-wiki.json")
#
# output = tokenizer.encode("Hello, y'all! How are you ?")
#
# print(output.tokens)
# # ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
# print(output.ids)
# # [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
#
#
import torch.nn

s = torch.nn.Embedding(3, 4)

print(s(torch.Tensor(torch.LongTensor([1, 2]))))