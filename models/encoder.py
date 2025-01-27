from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_features):
        super(Encoder, self).__init__()
        
        self.embedding_list = nn.ModuleList()

        for i in range(num_features):
            emb = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.embedding_list[i](x[:,i])

        return x_embedding
    