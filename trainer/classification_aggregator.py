import logging
import re

import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.core import ServerAggregator

from .classification_trainer import (
    is_router_param,
    get_client_loss_deltas,
    compute_router_weights,
    clear_client_loss_deltas,
)


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
    
    def aggregate(self, model_list, sample_num_list):
        """
        聚合多个客户端的模型参数。
        
        如果启用了 mor_adaptive_router_weight，对于 Router 参数使用基于
        epoch loss 差值的自适应加权平均，其他参数使用标准 FedAvg 加权平均。
        
        Args:
            model_list: 客户端模型参数列表，每项为 (sample_num, model_state_dict)
            sample_num_list: 样本数量列表 (可能未使用，取决于 model_list 格式)
        """
        logging.info("=" * 60)
        logging.info("[MoR Aggregator] aggregate() method CALLED")
        logging.info(f"[MoR Aggregator] model_list length: {len(model_list)}")
        logging.info("=" * 60)
        
        args = self.args
        # 支持两种配置访问方式
        adaptive_router = getattr(args, 'mor_adaptive_router_weight', None)
        if adaptive_router is None:
            mor_args = getattr(args, 'mor_args', {})
            if isinstance(mor_args, dict):
                adaptive_router = mor_args.get('mor_adaptive_router_weight', False)
            else:
                adaptive_router = getattr(mor_args, 'mor_adaptive_router_weight', False)
        
        logging.info(f"[MoR Aggregator] mor_adaptive_router_weight = {adaptive_router}")
        
        if not adaptive_router:
            # 使用默认的 FedAvg 聚合
            logging.info("[MoR Aggregator] Using default FedAvg aggregation")
            return self._default_aggregate(model_list)
        
        # 获取客户端 loss 差值
        loss_deltas = get_client_loss_deltas()
        logging.info(f"[MoR Adaptive] Client loss deltas: {loss_deltas}")
        
        # 如果没有 loss 差值记录，退化为默认 FedAvg
        if not loss_deltas:
            logging.warning("[MoR Adaptive] No loss deltas recorded, falling back to FedAvg")
            return self._default_aggregate(model_list)
        
        # 计算 Router 聚合权重
        client_ids = list(loss_deltas.keys())
        router_weights = compute_router_weights(client_ids)
        
        # 打印详细的加权信息
        logging.info("=" * 60)
        logging.info("[MoR Adaptive] ADAPTIVE ROUTER AGGREGATION ACTIVATED!")
        logging.info(f"[MoR Adaptive] Participating clients: {client_ids}")
        for cid in client_ids:
            delta = loss_deltas.get(cid, 'N/A')
            weight = router_weights.get(cid, 0.0)
            logging.info(f"  Client {cid}: loss_delta={delta:.6f}, weight={weight:.4f}")
        logging.info("=" * 60)
        
        # 开始聚合
        aggregated_params = {}
        
        # 获取所有参数名
        first_model = model_list[0][1] if isinstance(model_list[0], tuple) else model_list[0]
        param_names = list(first_model.keys())
        
        # 统计 Router 参数
        router_param_names = [p for p in param_names if is_router_param(p)]
        logging.info(f"[MoR Adaptive] Found {len(router_param_names)} router params: {router_param_names}")
        
        # 计算总样本数用于非 Router 参数的 FedAvg
        total_sample_num = sum([item[0] if isinstance(item, tuple) else 1 for item in model_list])
        
        for param_name in param_names:
            if is_router_param(param_name):
                # Router 参数：使用基于 loss 差值的加权平均
                aggregated_param = self._weighted_aggregate_router_param(
                    model_list, param_name, router_weights, client_ids
                )
            else:
                # 非 Router 参数：使用标准 FedAvg 加权
                aggregated_param = self._fedavg_aggregate_param(
                    model_list, param_name, total_sample_num
                )
            
            if aggregated_param is not None:
                aggregated_params[param_name] = aggregated_param
        
        # 清理本轮的 loss 差值记录
        clear_client_loss_deltas()
        
        logging.info("[MoR Adaptive] Aggregation complete with adaptive router weights")
        return aggregated_params
    
    def _default_aggregate(self, model_list):
        """默认的 FedAvg 聚合"""
        first_model = model_list[0][1] if isinstance(model_list[0], tuple) else model_list[0]
        aggregated_params = {}
        
        total_sample_num = sum([item[0] if isinstance(item, tuple) else 1 for item in model_list])
        
        for param_name in first_model.keys():
            aggregated_params[param_name] = self._fedavg_aggregate_param(
                model_list, param_name, total_sample_num
            )
        
        return aggregated_params
    
    def _fedavg_aggregate_param(self, model_list, param_name, total_sample_num):
        """FedAvg 方式聚合单个参数"""
        agg_param = None
        for item in model_list:
            if isinstance(item, tuple):
                sample_num, model_params = item
            else:
                sample_num = 1
                model_params = item
            
            if param_name not in model_params:
                continue
            
            param = model_params[param_name]
            weight = sample_num / total_sample_num
            
            if agg_param is None:
                agg_param = param * weight
            else:
                agg_param = agg_param + param * weight
        
        return agg_param
    
    def _weighted_aggregate_router_param(self, model_list, param_name, router_weights, client_ids):
        """
        使用自适应权重聚合 Router 参数。
        
        注意：这里假设 model_list 的顺序与 client_ids 对应。
        """
        agg_param = None
        valid_count = 0
        
        for idx, item in enumerate(model_list):
            if isinstance(item, tuple):
                _, model_params = item
            else:
                model_params = item
            
            if param_name not in model_params:
                continue
            
            # 获取对应客户端的权重
            if idx < len(client_ids):
                client_id = client_ids[idx]
                weight = router_weights.get(client_id, 0.0)
            else:
                # 如果没有对应的权重，使用均匀权重
                weight = 1.0 / len(model_list)
            
            param = model_params[param_name]
            
            if agg_param is None:
                agg_param = param * weight
            else:
                agg_param = agg_param + param * weight
            
            valid_count += 1
        
        # 如果部分客户端没有该参数，重新归一化
        if valid_count > 0 and valid_count < len(model_list):
            logging.info(f"[MoR Adaptive] Param {param_name}: only {valid_count}/{len(model_list)} clients have this param")
        
        return agg_param

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
