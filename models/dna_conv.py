import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_scatter import scatter


class DNAConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, num_pre_layers=1, num_post_layers=1, **kwargs):
        super(DNAConv, self).__init__(aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        if edge_dim is not None:
            self.pre_nn = MLP(in_channels=2 * in_channels + edge_dim, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)
        else:
            self.pre_nn = MLP(in_channels=2 * in_channels, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)

        self.lin_aggr = nn.Linear(in_channels, in_channels)
        self.post_nn = MLP(in_channels=in_channels, hidden_channels=out_channels,
                           out_channels=out_channels, num_layers=num_post_layers)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.pre_nn.reset_parameters()
        self.lin_aggr.reset_parameters()
        self.post_nn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return self.post_nn(out)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        return self.pre_nn(h)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        outs = self.lin_aggr(inputs)
        out = scatter(outs, index, dim=0, dim_size=dim_size, reduce='sum')

        return out
