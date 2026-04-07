import logging
import torch

import fedml
from data.data_loader import load
from fedml import FedMLRunner
from model.bert_model import BertForSequenceClassification
from model.distilbert_model import DistilBertForSequenceClassification
from model.llama_model import LlamaForSequenceClassification
from model.moe_llama_model import MoELlamaForSequenceClassification
from model.mor_llama_model import MoRLlamaForSequenceClassification, create_mor_config
from trainer.classification_aggregator import ClassificationAggregator
from trainer.classification_trainer import MyModelTrainer as MyCLSTrainer
from trainer.fedmoe_aggregator import FedMoEAggregator
from trainer.fedmoe_trainer import FedMoEModelTrainer
from trainer.fedmoe_utils import (
    aggregate_fedmoe_params,
    get_expert_id_from_param_name,
    get_fedmoe_config,
    is_fedmoe_enabled,
)
from transformers import (
    BertConfig,
    DistilBertConfig,
    LlamaConfig,
)


def create_model(args, output_dim=1):
    model_name = args.model
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification),
            "llama": (LlamaConfig, LlamaForSequenceClassification),
            "mor_llama": (LlamaConfig, MoRLlamaForSequenceClassification),
            "moe_llama": (LlamaConfig, MoELlamaForSequenceClassification),
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        },
    }
    try:
        config_class, model_class = MODEL_CLASSES[args.formulation][args.model_type]
    except KeyError:
        raise Exception("such model or formulation does not exist currently!")
    model_args = {}

    model_args["num_labels"] = output_dim
    config = config_class.from_pretrained(args.model, **model_args)
    
    # Handle LLaMA-based models specially
    if args.model_type == "llama":
        # Base LLaMA model (baseline without MoR)
        model = model_class(
            config=config,
            num_labels=output_dim,
        )
        
        # Load pretrained weights if specified
        if hasattr(args, 'model') and args.model:
            from transformers import LlamaForCausalLM
            try:
                pretrained = LlamaForCausalLM.from_pretrained(args.model)
                model.llama.load_state_dict(pretrained.state_dict(), strict=False)
                logging.info(f"Loaded pretrained LLaMA weights from {args.model}")
            except Exception as e:
                logging.warning(f"Could not load pretrained weights: {e}")
        
        # Handle fp16
        if getattr(args, 'fp16', False):
            logging.info("FP16 enabled - will use Automatic Mixed Precision (AMP)")
    
    elif args.model_type == "mor_llama":
        # MoR LLaMA model
        # Create MoR config from args
        mor_config = create_mor_config(args)
        
        # Get MoR loss coefficient
        mor_loss_coeff = getattr(args, 'mor_loss_coeff', 0.01)
        
        # Create MoR model
        model = model_class(
            config=config, 
            num_labels=output_dim, 
            mor_config=mor_config,
            mor_loss_coeff=mor_loss_coeff,
        )
        
        # Load pretrained weights if specified
        if hasattr(args, 'model') and args.model:
            # Import using the same path setup as mor_llama_model
            import os
            import sys
            _mor_path = os.path.join(os.path.dirname(__file__), 'model', 'mor')
            if _mor_path not in sys.path:
                sys.path.insert(0, _mor_path)
            from model.mor.model_mor.mor_model.modeling_llama import MoRLlamaForCausalLM
            try:
                pretrained = MoRLlamaForCausalLM.from_pretrained(args.model)
                model.mor_llama.load_state_dict(pretrained.state_dict(), strict=False)
                logging.info(f"Loaded pretrained weights from {args.model}")
            except Exception as e:
                logging.warning(f"Could not load MoR pretrained weights, fallback to base LLaMA: {e}")
                try:
                    from transformers import LlamaForCausalLM
                    pretrained = LlamaForCausalLM.from_pretrained(args.model)
                    model.mor_llama.load_state_dict(pretrained.state_dict(), strict=False)
                    logging.info(f"Loaded base LLaMA pretrained weights from {args.model}")
                except Exception as e2:
                    logging.warning(f"Could not load pretrained weights: {e2}")
        
        # Setup MoR architecture if enabled
        if getattr(args, 'mor_enable', False):
            model.setup_mor(mor_config)
            logging.info("MoR architecture setup complete")
        
        # Note: For fp16, we now use AMP (autocast + GradScaler) in trainer
        # instead of model.half() for better stability and automatic dtype handling
        if getattr(args, 'fp16', False):
            logging.info("FP16 enabled - will use Automatic Mixed Precision (AMP)")

    elif args.model_type == "moe_llama":
        # True MoE LLaMA model (independent LLaMA experts + sparse router)
        num_experts = int(getattr(args, 'moe_num_experts', 4))
        top_k = int(getattr(args, 'moe_top_k', 1))
        router_temperature = float(getattr(args, 'moe_router_temperature', 1.0))
        load_balance_loss_coef = float(getattr(args, 'moe_load_balance_loss_coef', 0.0))

        model = model_class(
            config=config,
            num_labels=output_dim,
            num_experts=num_experts,
            top_k=top_k,
            router_temperature=router_temperature,
            load_balance_loss_coef=load_balance_loss_coef,
        )

        # Load pretrained LLaMA weights to every expert
        if hasattr(args, 'model') and args.model:
            try:
                model.load_pretrained_experts(args.model)
                logging.info(
                    f"Loaded pretrained LLaMA weights into {num_experts} MoE experts from {args.model}"
                )
            except Exception as e:
                logging.warning(f"Could not load pretrained weights for MoE experts: {e}")

        if getattr(args, 'fp16', False):
            logging.info("FP16 enabled - will use Automatic Mixed Precision (AMP)")
    else:
        model = model_class.from_pretrained(args.model, config=config)
    
    if is_fedmoe_enabled(args):
        trainer = FedMoEModelTrainer(model, args)
    else:
        trainer = MyCLSTrainer(model, args)

    # calculate the model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    logging.info('model size: {:.3f}MB'.format(size_all_mb))

    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load(args)

    # load model and trainer
    model, trainer = create_model(args, output_dim)
    if is_fedmoe_enabled(args):
        aggregator = FedMoEAggregator(model, args)
    else:
        aggregator = ClassificationAggregator(model, args)
    
    # --- PATCH: force FedML SP to use our custom trainer ---
    import fedml.ml.trainer.my_model_trainer_classification as fedml_default_trainer
    from trainer.classification_trainer import MyModelTrainer as MyCLSTrainer
    from trainer.classification_trainer import is_router_param, get_trained_clients, mark_client_trained

    _orig_cls = fedml_default_trainer.ModelTrainerCLS

    class ModelTrainerCLS(_orig_cls):
        def train(self, train_data, device, args):
            trainer_cls = FedMoEModelTrainer if is_fedmoe_enabled(args) else MyCLSTrainer
            t = trainer_cls(self.model, args)
            t.id = getattr(self, "client_index", getattr(self, "id", 0))
            return t.train(train_data, device, args)

        def test(self, test_data, device, args):
            trainer_cls = FedMoEModelTrainer if is_fedmoe_enabled(args) else MyCLSTrainer
            t = trainer_cls(self.model, args)
            t.id = getattr(self, "client_index", getattr(self, "id", 0))
            # 补上可能缺失的属性
            if not hasattr(args, "round_idx"):
                args.round_idx = getattr(args, "round_idx", 0)
            if not hasattr(args, "rank"):
                args.rank = 0
            return t.test(test_data, device, args)

        def get_model_params(self):
            if is_fedmoe_enabled(args):
                t = FedMoEModelTrainer(self.model, args)
                t.id = getattr(self, "client_index", getattr(self, "id", 0))
                return t.get_model_params()

            """
            获取模型参数用于上传聚合。
            
            如果启用了 mor_skip_router_first_round，且当前客户端是第一次被训练，
            则排除 Router 相关参数，不参与聚合。
            """
            state_dict = self.model.cpu().state_dict()
            
            # 检查是否启用跳过首次训练router聚合
            skip_router_first = getattr(args, 'mor_skip_router_first_round', False)
            
            if not skip_router_first:
                return state_dict
            
            # 获取客户端ID
            client_id = getattr(self, 'client_index', getattr(self, 'id', None))
            trained_clients = get_trained_clients()
            
            if client_id is None:
                logging.warning("[MoR] Cannot determine client_id, returning full state_dict")
                return state_dict
            
            # 检查是否是首次训练
            is_first_round = client_id not in trained_clients
            
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
            if is_fedmoe_enabled(args):
                t = FedMoEModelTrainer(self.model, args)
                t.id = getattr(self, "client_index", getattr(self, "id", 0))
                t.set_model_params(model_parameters)
                return

            """设置模型参数，支持部分参数更新"""
            self.model.load_state_dict(model_parameters, strict=False)

    fedml_default_trainer.ModelTrainerCLS = ModelTrainerCLS
    logging.info("FedML SP ModelTrainerCLS patched to use custom MyModelTrainer.")
    # --- PATCH END ---
    
    # --- PATCH: force FedML SP to use our custom aggregator ---
    try:
        # FedML SP 模式使用 FedAvgAPI 类
        from fedml.simulation.sp.fedavg import FedAvgAPI
        from trainer.classification_trainer import (
            get_client_loss_deltas,
            compute_router_weights,
            clear_client_loss_deltas,
            is_router_param,
        )

        def _get_mor_config_value(key, default=None):
            value = getattr(args, key, None)
            if value is not None:
                return value
            mor_args = getattr(args, 'mor_args', {})
            if isinstance(mor_args, dict):
                return mor_args.get(key, default)
            return getattr(mor_args, key, default)

        def _resolve_router_tau_for_round():
            tau_const = float(_get_mor_config_value('mor_router_weight_tau', 1.0))
            tau_warmup = bool(_get_mor_config_value('mor_router_weight_tau_warmup', False))
            if not tau_warmup:
                return max(tau_const, 1e-8), "constant"

            tau_start = float(_get_mor_config_value('mor_router_weight_tau_warmup_start', 2.0))
            tau_end = float(_get_mor_config_value('mor_router_weight_tau_warmup_end', tau_const))
            warmup_rounds = int(_get_mor_config_value('mor_router_weight_tau_warmup_rounds', 10))
            round_idx = int(getattr(args, 'round_idx', 0))

            if warmup_rounds <= 0:
                tau = tau_end
            else:
                progress = min(max(round_idx, 0), warmup_rounds) / float(warmup_rounds)
                tau = tau_start + (tau_end - tau_start) * progress

            return max(float(tau), 1e-8), (
                f"warmup(start={tau_start}, end={tau_end}, rounds={warmup_rounds}, round_idx={round_idx})"
            )

        def _compute_fedmoe_param_delta_summary(previous_params, aggregated_params, num_experts):
            if previous_params is None:
                return None

            shared_sq_sum = 0.0
            expert_sq_sum = {expert_id: 0.0 for expert_id in range(num_experts)}
            shared_count = 0
            expert_count = {expert_id: 0 for expert_id in range(num_experts)}

            for name, new_param in aggregated_params.items():
                old_param = previous_params.get(name, None)
                if old_param is None or not torch.is_tensor(new_param) or not torch.is_tensor(old_param):
                    continue
                if new_param.shape != old_param.shape:
                    continue

                delta_sq = float(torch.sum((new_param.float() - old_param.float()) ** 2).item())
                expert_id = get_expert_id_from_param_name(name)
                if expert_id is None:
                    shared_sq_sum += delta_sq
                    shared_count += 1
                elif 0 <= expert_id < num_experts:
                    expert_sq_sum[expert_id] += delta_sq
                    expert_count[expert_id] += 1

            return {
                "shared_l2": shared_sq_sum ** 0.5,
                "shared_param_count": shared_count,
                "expert_l2": {k: v ** 0.5 for k, v in expert_sq_sum.items()},
                "expert_param_count": expert_count,
            }
        
        # 保存原始的 _aggregate 方法
        _orig_aggregate = FedAvgAPI._aggregate
        
        def _custom_aggregate(self, w_locals):
            """
            自定义聚合方法，使用 ClassificationAggregator 的逻辑。
            替换 FedML 默认的 FedAvgAPI._aggregate 方法。
            
            Args:
                w_locals: list of (sample_num, model_params) tuples
            """
            logging.info("=" * 60)
            logging.info("[MoR Aggregator PATCH] Custom _aggregate() method CALLED")
            logging.info(f"[MoR Aggregator PATCH] Received {len(w_locals)} client models")
            logging.info("=" * 60)
            
            if len(w_locals) == 0:
                logging.warning("[MoR Aggregator PATCH] No models to aggregate!")
                return None

            if is_fedmoe_enabled(args):
                cfg = get_fedmoe_config(args)
                prev_global_params = None
                if hasattr(self, "model_trainer") and hasattr(self.model_trainer, "model"):
                    prev_global_params = self.model_trainer.model.cpu().state_dict()

                aggregated_params, stats = aggregate_fedmoe_params(
                    w_locals=w_locals,
                    previous_global_params=prev_global_params,
                    num_experts=cfg["num_experts"],
                    keep_expert_if_no_contributor=cfg["keep_expert_if_no_contributor"],
                )
                delta_summary = _compute_fedmoe_param_delta_summary(
                    prev_global_params,
                    aggregated_params,
                    cfg["num_experts"],
                )
                logging.info(
                    "[FedMoE Aggregator PATCH] modular aggregation enabled. "
                    "shared_param_count=%s, expert_contributors=%s",
                    stats["shared_param_count"],
                    stats["expert_contributors"],
                )
                if delta_summary is not None:
                    logging.info(
                        "[FedMoE Aggregator PATCH] param_delta_l2 shared=%.6f (count=%s), experts=%s, expert_param_count=%s",
                        delta_summary["shared_l2"],
                        delta_summary["shared_param_count"],
                        delta_summary["expert_l2"],
                        delta_summary["expert_param_count"],
                    )
                return aggregated_params
            
            # 检查是否启用自适应 Router 聚合权重
            adaptive_router = bool(_get_mor_config_value('mor_adaptive_router_weight', False))
            
            logging.info(f"[MoR Aggregator PATCH] mor_adaptive_router_weight = {adaptive_router}")
            
            if not adaptive_router:
                # 使用默认 FedAvg 聚合
                logging.info("[MoR Aggregator PATCH] Using default FedAvg aggregation")
                return _fedavg_aggregate(w_locals)
            else:
                # 使用自适应 Router 聚合
                loss_deltas = get_client_loss_deltas()
                logging.info(f"[MoR Adaptive PATCH] Client loss deltas: {loss_deltas}")
                
                if not loss_deltas:
                    logging.warning("[MoR Adaptive PATCH] No loss deltas, falling back to FedAvg")
                    return _fedavg_aggregate(w_locals)
                else:
                    client_ids = list(loss_deltas.keys())
                    
                    # 读取权重计算模式配置
                    # mor_router_weight_mode: "absolute" (按loss绝对差值) 或 "relative" (按相对比例)
                    # 兼容旧键 mor_adaptive_weight_mode
                    weight_mode = _get_mor_config_value('mor_router_weight_mode', None)
                    if weight_mode is None:
                        weight_mode = _get_mor_config_value('mor_adaptive_weight_mode', 'absolute')

                    router_tau, tau_source = _resolve_router_tau_for_round()

                    logging.info(f"[MoR Adaptive PATCH] weight_mode = {weight_mode}, tau={router_tau:.6f} ({tau_source})")
                    router_weights = compute_router_weights(
                        client_ids,
                        weight_mode=weight_mode,
                        tau=router_tau,
                    )
                    
                    logging.info("=" * 60)
                    logging.info("[MoR Adaptive PATCH] ADAPTIVE ROUTER AGGREGATION ACTIVATED!")
                    logging.info(f"[MoR Adaptive PATCH] Weight mode: {weight_mode}, tau={router_tau:.6f}")
                    for cid in client_ids:
                        delta = loss_deltas.get(cid, 'N/A')
                        weight = router_weights.get(cid, 0.0)
                        logging.info(f"  Client {cid}: loss_delta={delta:.6f}, weight={weight:.4f}")
                    logging.info("=" * 60)
                    
                    result = _adaptive_aggregate(w_locals, router_weights, client_ids)
                    clear_client_loss_deltas()
                    return result
        
        def _fedavg_aggregate(w_locals):
            """标准 FedAvg 聚合"""
            (_, first_model) = w_locals[0]
            aggregated_params = {}
            total_sample_num = sum([item[0] for item in w_locals])
            
            for param_name in first_model.keys():
                agg_param = None
                for sample_num, model_params in w_locals:
                    if param_name not in model_params:
                        continue
                    param = model_params[param_name]
                    weight = sample_num / total_sample_num
                    if agg_param is None:
                        agg_param = param * weight
                    else:
                        agg_param = agg_param + param * weight
                if agg_param is not None:
                    aggregated_params[param_name] = agg_param
            
            return aggregated_params
        
        def _adaptive_aggregate(w_locals, router_weights, client_ids):
            """自适应 Router 权重聚合"""
            (_, first_model) = w_locals[0]
            aggregated_params = {}
            total_sample_num = sum([item[0] for item in w_locals])
            
            # 统计 Router 参数
            router_param_names = [p for p in first_model.keys() if is_router_param(p)]
            logging.info(f"[MoR Adaptive PATCH] Found {len(router_param_names)} router params")
            
            for param_name in first_model.keys():
                # 所有参数都用 router_weights 加权聚合
                agg_param = None
                for idx, (sample_num, model_params) in enumerate(w_locals):
                    if param_name not in model_params:
                        continue
                    if idx < len(client_ids):
                        weight = router_weights.get(client_ids[idx], 1.0 / len(w_locals))
                    else:
                        weight = 1.0 / len(w_locals)
                    param = model_params[param_name]
                    if agg_param is None:
                        agg_param = param * weight
                    else:
                        agg_param = agg_param + param * weight
                if agg_param is not None:
                    aggregated_params[param_name] = agg_param
            return aggregated_params
        
        # 替换 FedAvgAPI 的 _aggregate 方法
        FedAvgAPI._aggregate = _custom_aggregate
        logging.info("FedML SP FedAvgAPI._aggregate patched to use custom aggregator logic.")
    except ImportError as e:
        logging.warning(f"Could not patch FedML aggregator: {e}")
    # --- AGGREGATOR PATCH END ---
    
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
