from torch_geometric.transforms import BaseTransform

class ZINCTransform(BaseTransform):
    def __call__(self, data):
        data.y = data.y.unsqueeze(dim=1)
        return data
