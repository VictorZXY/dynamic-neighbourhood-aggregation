from torch_geometric.transforms import BaseTransform


class RemoveEdgeAttr(BaseTransform):
    def __call__(self, data):
        data.edge_attr = None
        return data


class UnsqueezeTargetDim(BaseTransform):
    def __call__(self, data):
        data.y = data.y.unsqueeze(dim=1)
        return data
