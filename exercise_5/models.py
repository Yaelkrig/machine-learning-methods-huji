import torch
import torch.nn as nn
import torch.nn.functional as F


class MySelfAttention(nn.Module):
    """
    Self attention layer
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The feature dimension the input tokens (d).
        """
        super(MySelfAttention, self).__init__()
        self.input_dim = input_dim
        ### YOUR CODE HERE ###
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        ### YOUR CODE HERE ###
        batch_size, seq_len, input_dim = x.size()

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (input_dim ** 0.5)

        attention = F.softmax(attention_scores, dim=2)
        attention_output = torch.bmm(attention, values)

        return attention_output


class MyLayerNorm(nn.Module):
    """
    Layer Normalization layer.
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The dimension of the input (T, d).
        """
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(*input_dim))
        self.beta = nn.Parameter(torch.zeros(*input_dim))
        self.eps = 1e-8

    def forward(self, x):
        ### YOUR CODE HERE ###
        mue = x.mean(dim=(1, 2), keepdim=True)
        sigma_powered = ((x-mue) ** 2).mean(dim=(1, 2), keepdim=True)

        normalized = (x - mue) / torch.sqrt(sigma_powered + self.eps)

        return self.gamma * normalized + self.beta


class MyTransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, max_len, input_dim):
        super(MyTransformerBlock, self).__init__()
        self.attention = MySelfAttention(input_dim)
        self.norm1 = MyLayerNorm((max_len, input_dim))
        self.norm2 = MyLayerNorm((max_len, input_dim))
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.attention(x)
        x = self.norm1(self.dropout(out) + x)
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.norm2(out + x)
        return out

class MyTransformer(nn.Module):
    """
    Transformer.
    """
    def __init__(self, vocab, max_len, num_of_blocks):
        """
        :param vocab: The vocabulary object.
        :param num_of_blocks: The number of transformer blocks.
        """
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = self.embedding.embedding_dim
        self.max_len = max_len
        self.blocks = nn.ModuleList([MyTransformerBlock(self.max_len, self.emb_dim) for _ in range(num_of_blocks)])
        self.fc = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        avg_pooling = x.mean(dim=1)
        x = self.fc(avg_pooling)
        return x

