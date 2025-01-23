import torch


class Logger:
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, eval_metric, log_path=None):
        self.eval_metric = eval_metric
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
            argmax = result['val'].argmax().item()
            print(f"Run {(run + 1):02d}: "
                  f"Best train {self.eval_metric}: {result['train'].max().item():.4f}, "
                  f"Best val {self.eval_metric}: {result['val'].max().item():.4f}, "
                  f"Final train {self.eval_metric}: {result['train'][argmax].item():.4f}, "
                  f"Final test {self.eval_metric}: {result['test'][argmax].item():.4f}")
        else:
            results = [{k: torch.tensor(v) for k, v in result.items()} for result in self.results]

            best_results = {'best_train': [], 'best_val': [], 'final_train': [], 'final_test': []}
            for result in results:
                argmax = result['val'].argmax().item()
                best_results['best_train'].append(result['train'].max().item())
                best_results['best_val'].append(result['val'].max().item())
                best_results['final_train'].append(result['train'][argmax].item())
                best_results['final_test'].append(result['test'][argmax].item())

            best_results = {k: torch.tensor(v) for k, v in best_results.items()}
            print("All runs:")
            best_train = best_results['best_train']
            print(f"Best train {self.eval_metric}: {best_train.mean():.4f} ± {best_train.std():.4f}")
            best_val = best_results['best_val']
            print(f"Best val {self.eval_metric}: {best_val.mean():.4f} ± {best_val.std():.4f}")
            final_train = best_results['final_train']
            print(f"Final train {self.eval_metric}: {final_train.mean():.4f} ± {final_train.std():.4f}")
            final_test = best_results['final_test']
            print(f"Final test {self.eval_metric}: {final_test.mean():.4f} ± {final_test.std():.4f}")
