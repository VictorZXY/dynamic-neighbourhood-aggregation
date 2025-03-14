import torch


class Logger:
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, eval_metric, log_path=None):
        self.eval_metric = eval_metric
        if eval_metric.lower() in ['mae']:
            self.mode = 'min'
        else:
            self.mode = 'max'
        self.log_path = log_path
        self.results = [{'train': [], 'val': [], 'test': []} for _ in range(runs)]

    def add_result(self, run, train_result, val_result, test_result):
        assert 0 <= run < len(self.results)
        self.results[run]['train'].append(train_result)
        self.results[run]['val'].append(val_result)
        self.results[run]['test'].append(test_result)

    def print_statistics(self, run=None):
        if run is not None:
            result = {k: torch.tensor(v) for k, v in self.results[run].items()}
            if self.mode == 'max':
                argmax = result['val'].argmax().item()
                print(f"Run {(run + 1):02d}: "
                    f"Best train {self.eval_metric}: {result['train'].max().item():.4f}, "
                    f"Best val {self.eval_metric}: {result['val'].max().item():.4f}, "
                    f"Best test {self.eval_metric}: {result['test'].max().item():.4f}")
            else:  
                argmin = result['val'].argmin().item()
                print(f"Run {(run + 1):02d}: "
                    f"Best train {self.eval_metric}: {result['train'].min().item():.4f}, "
                    f"Best val {self.eval_metric}: {result['val'].min().item():.4f}, "
                    f"Best test {self.eval_metric}: {result['test'].min().item():.4f}")
        else:
            results = [{k: torch.tensor(v) for k, v in result.items()} for result in self.results]

            best_results = {'best_train': [], 'best_val': [], 'best_test': [],
                            'final_train': [], 'final_val': [], 'final_test': []}
            for result in results:
                if self.mode == 'max':
                    argmax = result['val'].argmax().item()
                    best_results['best_train'].append(result['train'].max().item())
                    best_results['best_val'].append(result['val'].max().item())
                    best_results['best_test'].append(result['test'].max().item())
                    # best_results['final_train'].append(result['train'][argmax].item())
                    # best_results['final_val'].append(result['val'][argmax].item())  # this should be equal to best_val
                    # best_results['final_test'].append(result['test'][argmax].item())
                else:
                    argmin = result['val'].argmin().item()
                    best_results['best_train'].append(result['train'].min().item())
                    best_results['best_val'].append(result['val'].min().item())
                    best_results['best_test'].append(result['test'].min().item())
                    # best_results['final_train'].append(result['train'][argmin].item())
                    # best_results['final_val'].append(result['val'][argmin].item())  # this should be equal to best_val
                    # best_results['final_test'].append(result['test'][argmin].item())

            best_results = {k: torch.tensor(v) for k, v in best_results.items()}
            print("All runs:")
            best_train = best_results['best_train']
            print(f"Best train {self.eval_metric}: {best_train.mean():.4f} ± {best_train.std():.4f}")
            best_val = best_results['best_val']
            print(f"Best val {self.eval_metric}: {best_val.mean():.4f} ± {best_val.std():.4f}")
            best_test = best_results['best_test']
            print(f"Best test {self.eval_metric}: {best_test.mean():.4f} ± {best_test.std():.4f}")
            # final_train = best_results['final_train']
            # print(f"Final train {self.eval_metric}: {final_train.mean():.4f} ± {final_train.std():.4f}")
            # final_val = best_results['final_val']
            # print(f"Final val {self.eval_metric}: {final_val.mean():.4f} ± {final_val.std():.4f}")
            # final_test = best_results['final_test']
            # print(f"Final test {self.eval_metric}: {final_test.mean():.4f} ± {final_test.std():.4f}")
