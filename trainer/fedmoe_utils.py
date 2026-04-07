import hashlib
import logging
import random
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


_EXPERT_PARAM_RE = re.compile(r"^experts\.(\d+)\.")


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_nested_arg(args, container_name: str, key: str, default=None):
    value = getattr(args, key, None)
    if value is not None:
        return value

    container = getattr(args, container_name, None)
    if isinstance(container, dict):
        return container.get(key, default)
    if container is None:
        return default
    return getattr(container, key, default)


def get_fedmoe_config(args) -> Dict[str, object]:
    num_experts = int(_get_nested_arg(args, "moe_args", "moe_num_experts", 4))
    experts_per_client_default = min(2, num_experts)
    experts_per_client = int(
        _get_nested_arg(
            args,
            "moe_args",
            "fedmoe_experts_per_client",
            experts_per_client_default,
        )
    )
    assignment_seed = int(
        _get_nested_arg(args, "moe_args", "fedmoe_assignment_seed", 0)
    )
    keep_expert_if_no_contributor = _to_bool(
        _get_nested_arg(
            args,
            "moe_args",
            "fedmoe_keep_expert_if_no_contributor",
            True,
        ),
        default=True,
    )
    fedmoe_enable = _to_bool(
        _get_nested_arg(args, "moe_args", "fedmoe_enable", False), default=False
    )

    if num_experts < 1:
        raise ValueError("fedmoe requires moe_num_experts >= 1")
    if experts_per_client < 1 or experts_per_client > num_experts:
        raise ValueError(
            "fedmoe_experts_per_client must satisfy 1 <= m <= moe_num_experts"
        )

    return {
        "fedmoe_enable": fedmoe_enable,
        "num_experts": num_experts,
        "experts_per_client": experts_per_client,
        "assignment_seed": assignment_seed,
        "keep_expert_if_no_contributor": keep_expert_if_no_contributor,
    }


def is_fedmoe_enabled(args) -> bool:
    try:
        cfg = get_fedmoe_config(args)
    except Exception as e:
        logging.warning("[FedMoE] Invalid config detected, disable FedMoE: %s", e)
        return False
    return getattr(args, "model_type", None) == "moe_llama" and bool(
        cfg["fedmoe_enable"]
    )


def get_expert_id_from_param_name(param_name: str) -> Optional[int]:
    match = _EXPERT_PARAM_RE.match(param_name)
    if match is None:
        return None
    return int(match.group(1))


def get_client_expert_assignment(
    client_id: int,
    num_experts: int,
    experts_per_client: int,
    assignment_seed: int,
) -> List[int]:
    token = f"{assignment_seed}:{client_id}".encode("utf-8")
    seed_digest = hashlib.sha256(token).hexdigest()
    seed_value = int(seed_digest[:16], 16)
    rng = random.Random(seed_value)
    return sorted(rng.sample(list(range(num_experts)), experts_per_client))


def filter_state_dict_for_experts(
    state_dict: Dict[str, object], assigned_experts: Iterable[int]
) -> Dict[str, object]:
    selected: Set[int] = set(assigned_experts)
    filtered = {}
    for param_name, param_value in state_dict.items():
        expert_id = get_expert_id_from_param_name(param_name)
        if expert_id is None or expert_id in selected:
            filtered[param_name] = param_value
    return filtered


def set_trainable_experts(model, assigned_experts: Iterable[int]) -> Tuple[int, int]:
    if not hasattr(model, "experts"):
        return 0, 0

    selected = set(assigned_experts)
    trainable = 0
    frozen = 0
    for expert_id, expert in enumerate(model.experts):
        should_train = expert_id in selected
        for param in expert.parameters():
            param.requires_grad = should_train
        if should_train:
            trainable += 1
        else:
            frozen += 1
    return trainable, frozen


def _weighted_average_for_param(contributors: List[Tuple[float, object]]):
    if not contributors:
        return None
    total_weight = sum(float(sample_num) for sample_num, _ in contributors)
    if total_weight <= 0:
        equal_weight = 1.0 / len(contributors)
        result = None
        for _, param in contributors:
            result = param * equal_weight if result is None else result + param * equal_weight
        return result

    result = None
    for sample_num, param in contributors:
        weight = float(sample_num) / total_weight
        result = param * weight if result is None else result + param * weight
    return result


def aggregate_fedmoe_params(
    w_locals: List[Tuple[float, Dict[str, object]]],
    previous_global_params: Optional[Dict[str, object]],
    num_experts: int,
    keep_expert_if_no_contributor: bool = True,
):
    if not w_locals:
        return {}, {"expert_contributors": {}, "shared_param_count": 0}

    shared_param_names: Set[str] = set()
    expert_param_names: Dict[int, Set[str]] = {i: set() for i in range(num_experts)}
    expert_contributors: Dict[int, int] = {i: 0 for i in range(num_experts)}

    for _, local_state in w_locals:
        local_expert_hit = set()
        for param_name in local_state.keys():
            expert_id = get_expert_id_from_param_name(param_name)
            if expert_id is None:
                shared_param_names.add(param_name)
            elif 0 <= expert_id < num_experts:
                expert_param_names[expert_id].add(param_name)
                local_expert_hit.add(expert_id)
        for expert_id in local_expert_hit:
            expert_contributors[expert_id] += 1

    aggregated_params: Dict[str, object] = {}

    for param_name in shared_param_names:
        contributors = []
        for sample_num, local_state in w_locals:
            if param_name in local_state:
                contributors.append((sample_num, local_state[param_name]))
        averaged = _weighted_average_for_param(contributors)
        if averaged is not None:
            aggregated_params[param_name] = averaged

    for expert_id in range(num_experts):
        for param_name in expert_param_names[expert_id]:
            contributors = []
            for sample_num, local_state in w_locals:
                if param_name in local_state:
                    contributors.append((sample_num, local_state[param_name]))
            averaged = _weighted_average_for_param(contributors)
            if averaged is not None:
                aggregated_params[param_name] = averaged

        if expert_contributors[expert_id] == 0 and keep_expert_if_no_contributor:
            if previous_global_params is None:
                continue
            for param_name, param_value in previous_global_params.items():
                if get_expert_id_from_param_name(param_name) == expert_id:
                    aggregated_params[param_name] = param_value

    if previous_global_params is not None:
        for param_name, param_value in previous_global_params.items():
            if param_name not in aggregated_params:
                expert_id = get_expert_id_from_param_name(param_name)
                if expert_id is None:
                    aggregated_params[param_name] = param_value

    stats = {
        "expert_contributors": expert_contributors,
        "shared_param_count": len(shared_param_names),
    }
    return aggregated_params, stats
