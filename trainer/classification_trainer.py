import torch
from torch.cuda.amp import autocast, GradScaler

from fedml.core import ClientTrainer
from fedml.model.nlp.model_args import ClassificationArgs
from .text_classification_utils import *


# 全局记录已训练过的客户端集合 (用于跳过首次训练的router聚合)
_trained_clients = set()

# 全局记录客户端的epoch loss差值 (最后epoch - 第一epoch)
# key: client_id, value: loss_delta (负值表示loss下降)
_client_loss_deltas = {}

# 全局记录客户端的 first_epoch_loss (用于计算相对比例)
# key: client_id, value: first_epoch_loss
_client_first_epoch_losses = {}


def get_trained_clients():
    """获取已训练过的客户端集合"""
    return _trained_clients


def mark_client_trained(client_id):
    """标记客户端已训练过"""
    _trained_clients.add(client_id)


def reset_trained_clients():
    """重置训练记录 (用于新实验)"""
    global _trained_clients
    _trained_clients = set()


def get_client_loss_deltas():
    """获取所有客户端的loss差值字典"""
    return _client_loss_deltas.copy()


def set_client_loss_delta(client_id, loss_delta, first_epoch_loss=None):
    """设置客户端的loss差值 (最后epoch loss - 第一epoch loss)"""
    _client_loss_deltas[client_id] = loss_delta
    if first_epoch_loss is not None:
        _client_first_epoch_losses[client_id] = first_epoch_loss


def get_client_first_epoch_losses():
    """获取所有客户端的first_epoch_loss字典"""
    return _client_first_epoch_losses.copy()


def clear_client_loss_deltas():
    """清空loss差值记录 (每轮聚合后调用)"""
    global _client_loss_deltas, _client_first_epoch_losses
    _client_loss_deltas = {}
    _client_first_epoch_losses = {}


def compute_router_weights(client_ids, weight_mode="absolute", tau=1.0):
    """
    根据客户端loss差值计算Router聚合权重。
    
    Args:
        client_ids: 当前轮次参与训练的客户端ID列表
        weight_mode: 权重计算模式
            - "absolute": 按 loss_delta 绝对量 (loss_end - loss_start)
            - "relative": 按相对比例 (loss_start - loss_end) / loss_start
        tau: softmax 温度参数 (tau > 0)
    
    Returns:
        dict: {client_id: weight} 归一化权重字典
    """
    import numpy as np
    
    # 获取参与本轮训练的客户端的loss差值
    scores = []
    valid_clients = []
    
    for cid in client_ids:
        if cid in _client_loss_deltas:
            delta = _client_loss_deltas[cid]
            
            if weight_mode == "relative":
                # 相对比例: (loss_start - loss_end) / loss_start
                # = -delta / loss_start (因为 delta = loss_end - loss_start)
                first_loss = _client_first_epoch_losses.get(cid, None)
                if first_loss is not None and first_loss > 1e-8:
                    # 相对下降比例，正值表示下降
                    relative_drop = -delta / first_loss
                    scores.append(relative_drop)
                    valid_clients.append(cid)
                    logging.info(f"[MoR Weight] Client {cid}: relative_drop={relative_drop:.6f} (delta={delta:.6f}, first_loss={first_loss:.6f})")
            else:
                # 绝对量: 直接用 -delta (loss下降越多，-delta越大)
                scores.append(-delta)
                valid_clients.append(cid)
    
    if not scores:
        # 无有效数据，返回均匀权重
        return {cid: 1.0 / len(client_ids) for cid in client_ids}
    
    # 应用 softmax 归一化
    scores = np.array(scores)

    # 温度缩放，tau 越小分布越尖锐，tau 越大分布越平滑
    try:
        tau = float(tau)
    except (TypeError, ValueError):
        logging.warning(f"[MoR Weight] Invalid tau={tau}, fallback to 1.0")
        tau = 1.0
    if tau <= 0:
        logging.warning(f"[MoR Weight] Non-positive tau={tau}, clamp to 1e-8")
        tau = 1e-8
    scores = scores / tau
    
    # 防止数值溢出
    scores = scores - np.max(scores)
    exp_weights = np.exp(scores)
    weights = exp_weights / np.sum(exp_weights)
    
    result = {cid: float(w) for cid, w in zip(valid_clients, weights)}
    
    # 对于没有loss记录的客户端，给予最小权重
    for cid in client_ids:
        if cid not in result:
            result[cid] = 0.0
    
    return result


def is_router_param(param_name):
    """判断参数名是否是Router相关参数"""
    # mor_router: 主路由器 (例如 mor_llama.model.layers.X.mor_router.router.weight)
    # mlp_router: 辅助路由器 (用于 aux_router 采样策略)
    # router_bias: 路由偏置 (用于 loss_free 平衡策略)
    router_keywords = ['mor_router', 'mlp_router', 'router_bias']
    return any(keyword in param_name for keyword in router_keywords)


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        """
        获取模型参数用于上传聚合。
        
        如果启用了 mor_skip_router_first_round，且当前客户端是第一次被训练，
        则排除 Router 相关参数，不参与聚合。
        """
        state_dict = self.model.cpu().state_dict()
        
        # 检查是否启用跳过首次训练router聚合
        args = getattr(self, 'args', None)
        skip_router_first = getattr(args, 'mor_skip_router_first_round', False) if args else False
        
        if not skip_router_first:
            return state_dict
        
        # 获取客户端ID
        client_id = getattr(self, 'id', getattr(self, 'client_index', None))
        
        if client_id is None:
            logging.warning("[MoR] Cannot determine client_id, returning full state_dict")
            return state_dict
        
        # 检查是否是首次训练
        is_first_round = client_id not in _trained_clients
        
        if is_first_round:
            # 首次训练：排除Router参数
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not is_router_param(k)
            }
            excluded_count = len(state_dict) - len(filtered_state_dict)
            logging.info(f"[MoR] Client {client_id} first training: excluding {excluded_count} router params from aggregation")
            
            # 标记客户端已训练过
            mark_client_trained(client_id)
            
            return filtered_state_dict
        else:
            # 非首次训练：上传所有参数包括Router
            logging.info(f"[MoR] Client {client_id} already trained before: uploading all params including router")
            return state_dict

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    @staticmethod
    def _is_finite_tensor(value):
        if value is None or not torch.is_tensor(value):
            return False
        return bool(torch.isfinite(value).all().item())

    @staticmethod
    def _extract_logits(model_output):
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        if hasattr(model_output, "logits"):
            return model_output.logits
        return model_output

    def _forward_and_compute_loss(self, model, x, attention_mask, labels, criterion, args):
        is_moe = getattr(args, "model_type", "") == "moe_llama"

        if getattr(args, "model_class", "") == "transformer":
            if is_moe:
                model_output = model(x, attention_mask=attention_mask, labels=labels)
            else:
                model_output = model(x, attention_mask=attention_mask)
        else:
            model_output = model(x)

        if is_moe:
            if isinstance(model_output, (tuple, list)):
                if len(model_output) < 2:
                    raise RuntimeError("moe_llama output must contain (loss, logits)")
                loss, logits = model_output[0], model_output[1]
            elif hasattr(model_output, "loss") and hasattr(model_output, "logits"):
                loss, logits = model_output.loss, model_output.logits
            else:
                raise RuntimeError("Unsupported moe_llama output format")
            return loss, logits

        logits = self._extract_logits(model_output)
        loss = criterion(logits, labels)
        return loss, logits

    def train(self, train_data, device, args, test_data=None):
        model_args = ClassificationArgs()
        model_args.model_name = args.model
        model_args.model_type = args.model_type
        # model_args.load(model_args.model_name)
        # model_args.num_labels = output_dim
        model_args.update_from_dict(
            {
                "fl_algorithm": args.federated_optimizer,
                "freeze_layers": args.freeze_layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "do_lower_case": args.do_lower_case,
                "manual_seed": args.random_seed,
                # for ignoring the cache features.
                "reprocess_input_data": args.reprocess_input_data,
                "overwrite_output_dir": True,
                "max_seq_length": args.max_seq_length,
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "evaluate_during_training": False,  # Disabled for FedAvg.
                "evaluate_during_training_steps": args.evaluate_during_training_steps,
                "fp16": args.fp16,
                "data_file_path": args.data_file_path,
                "partition_file_path": args.partition_file_path,
                "partition_method": args.partition_method,
                "dataset": args.dataset,
                "output_dir": args.output_dir,
                "is_debug_mode": args.is_debug_mode,
                "fedprox_mu": args.fedprox_mu,
                "optimizer": args.client_optimizer,
            }
        )
        model = self.model

        model.to(device)
        model.train()

        # Snapshot current global model parameters for manual proximal regularization.
        global_params = [p.detach().clone() for p in model.parameters()]
        
        # Initialize AMP components for fp16 training
        use_amp = getattr(args, 'fp16', False)
        scaler = GradScaler() if use_amp else None
        
        tr_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss().to(device)
        iteration_in_total = (
            len(train_data) // args.gradient_accumulation_steps * args.epochs
        )
        optimizer, scheduler = build_optimizer(model, iteration_in_total, model_args)
        epoch_loss = []
        total_skipped_non_finite_batches = 0

        is_moe = getattr(args, "model_type", "") == "moe_llama"
        moe_num_experts = int(getattr(model, "num_experts", 0)) if is_moe else 0

        model.zero_grad()
        for epoch in range(args.epochs):
            batch_loss = []
            accumulated_steps = 0
            epoch_skipped_non_finite_batches = 0
            epoch_router_entropy_sum = 0.0
            epoch_router_entropy_count = 0
            epoch_expert_hits = (
                torch.zeros(moe_num_experts, dtype=torch.long) if moe_num_experts > 0 else None
            )

            for batch_idx, batch in enumerate(train_data):
                x = batch[1].to(device)
                attention_mask = batch[2].to(device)
                labels = batch[4].to(device)

                with autocast(enabled=use_amp):
                    loss, logits = self._forward_and_compute_loss(
                        model=model,
                        x=x,
                        attention_mask=attention_mask,
                        labels=labels,
                        criterion=criterion,
                        args=args,
                    )

                    if not self._is_finite_tensor(loss) or not self._is_finite_tensor(logits):
                        epoch_skipped_non_finite_batches += 1
                        total_skipped_non_finite_batches += 1
                        logging.warning(
                            "[FedMoE Stability] Skip non-finite batch. client=%s epoch=%s batch=%s",
                            self.id,
                            epoch,
                            batch_idx,
                        )
                        model.zero_grad()
                        continue

                    mu = float(getattr(args, "fedprox_mu", 0.0))
                    if mu > 0.0:
                        proximal_term = 0.0
                        for local_param, global_param in zip(model.parameters(), global_params):
                            proximal_term += torch.square((local_param - global_param).norm(2))
                        loss = loss + (mu / 2.0) * proximal_term

                    raw_loss_for_log = loss
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += float(raw_loss_for_log.detach().item())
                accumulated_steps += 1

                if is_moe and hasattr(model, "get_last_router_stats"):
                    router_stats = model.get_last_router_stats()
                    if isinstance(router_stats, dict):
                        entropy = router_stats.get("entropy", None)
                        if isinstance(entropy, float):
                            epoch_router_entropy_sum += entropy
                            epoch_router_entropy_count += 1

                        hits = router_stats.get("expert_hits", None)
                        if epoch_expert_hits is not None and isinstance(hits, list) and len(hits) == len(epoch_expert_hits):
                            epoch_expert_hits += torch.tensor(hits, dtype=torch.long)

                if accumulated_steps % args.gradient_accumulation_steps == 0:
                    if args.clip_grad_norm:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    batch_loss.append(tr_loss)
                    tr_loss = 0.0

            if accumulated_steps % args.gradient_accumulation_steps != 0:
                if args.clip_grad_norm:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                model.zero_grad()
                batch_loss.append(tr_loss)
                tr_loss = 0.0

            epoch_avg = float("nan")
            if len(batch_loss) > 0:
                epoch_avg = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(epoch_avg)

            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, epoch_avg
                )
            )
            if is_moe:
                avg_entropy = (
                    epoch_router_entropy_sum / epoch_router_entropy_count
                    if epoch_router_entropy_count > 0
                    else float("nan")
                )
                expert_hit_log = epoch_expert_hits.tolist() if epoch_expert_hits is not None else []
                logging.info(
                    "[FedMoE Diagnostics] Client=%s Epoch=%s router_entropy=%.6f expert_hits=%s skipped_non_finite_batches=%s",
                    self.id,
                    epoch,
                    avg_entropy,
                    expert_hit_log,
                    epoch_skipped_non_finite_batches,
                )

            if args.evaluate_during_training and test_data is not None:
                metrics = self.test(test_data, device, args)
                logging.info(
                    "Client Index = {}\tEpoch: {}\tAccuracy: {:.6f}".format(
                        self.id, epoch, metrics["test_correct"] / metrics["test_total"]
                    )
                )

        if total_skipped_non_finite_batches > 0:
            logging.warning(
                "[FedMoE Stability] Client=%s total skipped non-finite batches=%s",
                self.id,
                total_skipped_non_finite_batches,
            )
        
        # 记录epoch loss差值用于Router自适应加权聚合
        # 检查配置 - 支持两种访问方式: args.mor_adaptive_router_weight 或 args.mor_args.mor_adaptive_router_weight
        adaptive_router_weight = getattr(args, 'mor_adaptive_router_weight', None)
        if adaptive_router_weight is None:
            mor_args = getattr(args, 'mor_args', {})
            if isinstance(mor_args, dict):
                adaptive_router_weight = mor_args.get('mor_adaptive_router_weight', False)
            else:
                adaptive_router_weight = getattr(mor_args, 'mor_adaptive_router_weight', False)
        
        logging.info(f"[MoR Adaptive DEBUG] mor_adaptive_router_weight = {adaptive_router_weight}, epoch_loss count = {len(epoch_loss)}")
        
        finite_epoch_loss = [
            v for v in epoch_loss if isinstance(v, float) and torch.isfinite(torch.tensor(v))
        ]
        if adaptive_router_weight and len(finite_epoch_loss) >= 2:
            first_epoch_loss = finite_epoch_loss[0]
            last_epoch_loss = finite_epoch_loss[-1]
            loss_delta = last_epoch_loss - first_epoch_loss  # 负值表示loss下降
            client_id = getattr(self, 'id', getattr(self, 'client_index', 0))
            set_client_loss_delta(client_id, loss_delta, first_epoch_loss=first_epoch_loss)
            logging.info(
                f"[MoR Adaptive] Client {client_id}: first_epoch_loss={first_epoch_loss:.6f}, "
                f"last_epoch_loss={last_epoch_loss:.6f}, delta={loss_delta:.6f}"
            )

    def test(self, test_data, device, args):
        rank = getattr(args, 'rank', 0)
        round_idx = getattr(args, 'round_idx', 0)
        logging.info(f"----------test_on_the_client {rank} @ round {round_idx}--------")
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        use_amp = getattr(args, 'fp16', False)
        skipped_non_finite_batches = 0

        criterion = torch.nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                if args.model_class == "transformer":
                    x = batch[1].to(device)
                    attention_mask = batch[2].to(device)
                    target = batch[4].to(device)
                    # Note: x (input_ids) should remain as LongTensor for embedding lookups
                else:
                    x, target = batch[0].to(device), batch[1].to(device)
                
                # Use autocast for inference as well when fp16 enabled
                with autocast(enabled=use_amp):
                    if args.model_class == "transformer":
                        pred = model(x, attention_mask=attention_mask)
                    else:
                        pred = model(x)
                    if args.model_class == "transformer":
                        pred = pred[0]
                    loss = criterion(pred, target)

                if not self._is_finite_tensor(pred) or not self._is_finite_tensor(loss):
                    skipped_non_finite_batches += 1
                    logging.warning(
                        "[FedMoE Stability] Skip non-finite eval batch. client=%s round=%s batch=%s",
                        rank,
                        round_idx,
                        batch_idx,
                    )
                    continue

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        
        # 计算准确率和平均损失
        accuracy = metrics["test_correct"] / metrics["test_total"] if metrics["test_total"] > 0 else 0.0
        avg_loss = metrics["test_loss"] / metrics["test_total"] if metrics["test_total"] > 0 else 0.0
        
        # 打印测试结果
        logging.info(f"Client {rank} @ Round {round_idx} Test Results:")
        logging.info(f"  Total samples: {metrics['test_total']}")
        logging.info(f"  Correct: {metrics['test_correct']}")  
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Average Loss: {avg_loss:.6f}")
        if skipped_non_finite_batches > 0:
            logging.warning(f"  Skipped non-finite eval batches: {skipped_non_finite_batches}")
        
        return metrics
