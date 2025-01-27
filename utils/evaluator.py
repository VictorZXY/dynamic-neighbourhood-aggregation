class ZINCEvaluator:
    def __init__(self, **kwargs):
        self.eval_metric = 'mae'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs) or (num_graphs, 1)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, 1)
        """
        
        if y_true.dim() == 1:
            total_error = (y_pred.squeeze() - y_true).abs().sum().item()
        else:
            total_error = (y_pred - y_true).abs().sum().item()

        return {self.eval_metric: total_error / len(y_true)}
        