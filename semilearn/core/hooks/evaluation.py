# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    def before_run(self, algorithm):
        eval_dict = algorithm.evaluate('eval')
        if 'out' in algorithm.loader_dict:
            test_dict = algorithm.evaluate('out')

        if 'test' in algorithm.loader_dict:
            # evaluate on test dataset
            test_dict = algorithm.evaluate('test')

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it


            if 'out' in algorithm.loader_dict:
                test_dict = algorithm.evaluate('out')
                algorithm.log_dict.update(test_dict)

                if algorithm.log_dict['out/top-1-acc'] > algorithm.best_out_acc:
                    algorithm.best_out_acc = algorithm.log_dict['out/top-1-acc']
                    algorithm.best_out_it = algorithm.it

            if 'test' in algorithm.loader_dict:
                # evaluate on test dataset
                test_dict = algorithm.evaluate('test')
                algorithm.log_dict.update(test_dict)

                if algorithm.log_dict['test/top-1-acc'] > algorithm.best_all_acc:
                    algorithm.best_all_acc = algorithm.log_dict['test/top-1-acc']
                    algorithm.best_all_it = algorithm.it



    
    def after_run(self, algorithm):
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        # if 'test' in algorithm.loader_dict:
        #     # load the best model and evaluate on test dataset
        #     best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
        #     algorithm.load_model(best_model_path)
        #     test_dict = algorithm.evaluate('test')
        #     results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict

        if 'out' in algorithm.loader_dict:
            out_dict = {'out/best_acc': algorithm.best_out_acc, 'out/best_it': algorithm.best_out_it}
            algorithm.results_dict.update(out_dict)
        if 'test' in algorithm.loader_dict:
            all_dict = {'all/best_acc': algorithm.best_all_acc, 'all/best_it': algorithm.best_all_it}
            algorithm.results_dict.update(all_dict)