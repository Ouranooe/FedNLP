import logging
import re

import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.core import ServerAggregator


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class ClassificationAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        """
        设置模型参数，支持部分参数更新。
        
        当启用 mor_skip_router_first_round 时，首次训练的客户端不上传router参数，
        因此需要使用 strict=False 来允许部分加载。
        """
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters, strict=False)

    def set_model_size(self):
        """
        Override to calculate MoR theoretical model size.
        For MoR models, shared parameters are only counted once.
        """
        try:
            args = self.args
            model = self.model
            
            if getattr(args, 'model_type', '') == "mor_llama" and hasattr(model, 'mor_llama'):
                # For MoR models, calculate theoretical size based on sharing strategy
                sharing_strategy = getattr(args, 'recursive_sharing', 'middle_cycle')
                num_recursion = getattr(args, 'recursive_num_recursion', 3)
                base_depth_arg = getattr(args, 'recursive_base_depth', None)
                
                num_hidden_layers = model.mor_llama.config.num_hidden_layers
                
                # Calculate base_depth following llama.py logic
                if base_depth_arg is not None and num_recursion == 1:
                    base_depth = base_depth_arg
                else:
                    if sharing_strategy in ["cycle", "sequence"]:
                        base_depth = int(num_hidden_layers // num_recursion)
                    elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
                        base_depth = int((num_hidden_layers - 2) // num_recursion)
                    else:
                        base_depth = num_hidden_layers
                
                # Calculate unique layer count
                if sharing_strategy in ["middle_cycle", "middle_sequence"]:
                    unique_layer_count = base_depth + 2
                else:
                    unique_layer_count = base_depth
                
                # Calculate theoretical parameter size
                param_size = 0
                
                # Non-layer parameters
                for name, param in model.named_parameters():
                    if '.layers.' not in name:
                        param_size += param.nelement() * param.element_size()
                
                # Unique layer indices
                unique_layer_indices = set()
                if sharing_strategy in ["middle_cycle", "middle_sequence"]:
                    unique_layer_indices.add(0)
                    unique_layer_indices.add(num_hidden_layers - 1)
                    for i in range(base_depth):
                        if sharing_strategy == "middle_cycle":
                            unique_layer_indices.add(1 + i)
                        else:
                            unique_layer_indices.add(1 + i * num_recursion)
                else:
                    for i in range(base_depth):
                        if sharing_strategy == "cycle":
                            unique_layer_indices.add(i)
                        else:
                            unique_layer_indices.add(i * num_recursion)
                
                # Layer parameters: only count unique layers
                for name, param in model.named_parameters():
                    if '.layers.' in name:
                        match = re.search(r'\.layers\.(\d+)\.', name)
                        if match:
                            layer_idx = int(match.group(1))
                            if layer_idx in unique_layer_indices:
                                param_size += param.nelement() * param.element_size()
                
                # Buffer size
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                
                self.model_size = (param_size + buffer_size) / 1024 ** 2
                logging.info(f'[Communication] MoR model theoretical size: {self.model_size:.2f} MB '
                             f'(sharing={sharing_strategy}, base_depth={base_depth}, '
                             f'unique_layers={unique_layer_count}/{num_hidden_layers})')
            else:
                # Default behavior for non-MoR models
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                
                self.model_size = (param_size + buffer_size) / 1024 ** 2
                logging.info(f'[Communication] Model size: {self.model_size:.2f} MB')
        except Exception as e:
            logging.warning(f'[Communication] Failed to calculate model size: {e}')
            # Fallback to default calculation
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            self.model_size = (param_size + buffer_size) / 1024 ** 2
            logging.info(f'[Communication] Model size (fallback): {self.model_size:.2f} MB')

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = torch.nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                x = batch[1].to(device)
                target = batch[4].to(device)
                # x = x.to(device)
                # target = target.to(device)
                pred = model(x)
                pred = pred[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info(f"----------test_on_the_server @ round {args.round_idx}--------")
        accuracy_list, loss_list, metric_list = [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics = self._test(test_data, device, args)
            metric_list.append(metrics)
            accuracy_list.append(metrics["test_correct"] / metrics["test_total"])
            loss_list.append(metrics["test_loss"] / metrics["test_total"])
            logging.info(
                "Client {}, Test accuracy = {}".format(
                    client_idx, metrics["test_correct"] / metrics["test_total"]
                )
            )
        avg_accuracy = np.mean(np.array(accuracy_list))
        avg_loss = np.mean(np.array(loss_list))
        logging.info("Test Accuracy = {}".format(avg_accuracy))
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": avg_accuracy, "round": self.args.round_idx})
            wandb.log({"Test/Loss": avg_loss, "round": self.args.round_idx})
        mlops.log({"round_idx": args.round_idx, "loss": avg_loss, "evaluation_result": avg_accuracy})
        return True
