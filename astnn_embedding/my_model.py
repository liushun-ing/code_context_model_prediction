import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
print(torch.__version__)
class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained embedding 这里设置的是根据训练word2vec时，生成的预训练权重
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        """将tensor送到GPU"""
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):  # encodes, [0, 1, 2,....., ?]
        size = len(node)  # ?
        if not size:
            return None
        batch_current = self.create_tensor(torch.Tensor(torch.zeros(size, self.embedding_dim)))  # ?*128
        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
            index.append(i)
            current_node.append(node[i][0])
            temp = node[i][1:]
            c_num = len(temp)
            for j in range(c_num):
                if temp[j][0] != -1:
                    if len(children_index) <= j:
                        children_index.append([i])
                        children.append([temp[j]])
                    else:
                        children_index[j].append(i)
                        children[j].append(temp[j])
        # embedding默认会随机生成词向量，这里应该是不需要的，设置种子为0可以针对相同输入输出固定词向量
        batch_current = self.W_c(batch_current.index_copy(0, torch.Tensor(self.th.LongTensor(index)),
                                                          self.embedding(
                                                              torch.Tensor(self.th.LongTensor(current_node)))))
        for c in range(len(children)):
            zeros = self.create_tensor(torch.Tensor(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, torch.Tensor(self.th.LongTensor(children_index[c])), tree)
        b_in = torch.Tensor(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):  # encodes ?
        self.batch_size = bs  # ?
        self.batch_node = self.create_tensor(torch.Tensor(torch.zeros(self.batch_size, self.encode_dim)))
        # ? * 128
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        # 将一系列张量沿着一个新的维度进行拼接，默认为0
        self.node_list = torch.stack(self.node_list)
        # ?? * ? *128  ??应该是表示对子树进行递归后生成的向量数目
        max_node = torch.max(self.node_list, 0)[0]  # ? * 128
        # 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
        return max_node


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True,
                 pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = torch.Tensor(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = torch.Tensor(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return torch.Tensor(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return torch.Tensor(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        """
        获取num * encode_dim的全零张量

        Args:
            num: 需要的行数

        Returns: 形状为(num, self.encode_dim)全零张量
        """
        zeros = torch.Tensor(torch.zeros(num, self.encode_dim))
        # 用于创建一个形状为(num, self.encode_dim)的全零张量。其中，num张量的行数；self.encode_dim张量的列数
        if self.gpu:
            return zeros.cuda()  # 将全零张量移到GPU,加快计算速度
        return zeros

    def encode(self, x):  # ? 应该是ast根节点子树个数
        """
        对ast sequence编码，返回最终的200维度向量

        Args:
            x: ast解析后的sequences

        Returns: 200维度向量表示
        """
        b = []
        b.append(x)
        x = b
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])
        # ?
        encodes = self.encoder(encodes, sum(lens))
        # ? * 128
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end])
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            start = end
        # 将多个张量沿着指定的维度连接起来
        encodes = torch.cat(seq)
        # 将 encodes 张量的形状从 (N, *) 变为 (self.batch_size, max_len, *)，其中 N 是原始张量的元素数量
        encodes = encodes.view(self.batch_size, max_len, -1)
        # 将填充后的序列数据转换为PackedSequence对象，以便在RNN中使用
        encodes = nn.utils.rnn.pack_padded_sequence(encodes, torch.LongTensor(lens), True, False)
        gru_out, _ = self.bigru(encodes, self.hidden)
        # 将填充后的序列数据还原为原始形状, 并返回还原后的序列和原始长度信息
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=-1e9)
        # 用于交换张量的两个维度。将gru_out张量的第1个和第2个维度进行交换。
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # 对输入的张量gru_out进行一维最大池化操作。
        # 具体来说，它将gru_out沿着第二个维度（即长度维度）进行最大池化，池化窗口的大小为gru_out.size(2)。
        # 最后，使用squeeze(2)方法将结果张量的第三个维度（即长度维度）压缩为1。
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = gru_out[0]
        return gru_out
