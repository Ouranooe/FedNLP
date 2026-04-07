import logging
from typing import Optional, Tuple

from .classification_trainer import MyModelTrainer
from .fedmoe_utils import (
    filter_state_dict_for_experts,
    get_client_expert_assignment,
    get_fedmoe_config,
    is_fedmoe_enabled,
    set_trainable_experts,
)


def _resolve_client_id(trainer_or_wrapper) -> Optional[int]:
    return getattr(
        trainer_or_wrapper, "id", getattr(trainer_or_wrapper, "client_index", None)
    )


class FedMoEModelTrainer(MyModelTrainer):
    def _resolve_assignment(self, args) -> Tuple[dict, Optional[int], Optional[list]]:
        cfg = get_fedmoe_config(args)
        client_id = _resolve_client_id(self)
        if client_id is None:
            return cfg, None, None
        assigned_experts = get_client_expert_assignment(
            client_id=client_id,
            num_experts=cfg["num_experts"],
            experts_per_client=cfg["experts_per_client"],
            assignment_seed=cfg["assignment_seed"],
        )
        return cfg, client_id, assigned_experts

    def get_model_params(self):
        args = getattr(self, "args", None)
        state_dict = self.model.cpu().state_dict()
        if args is None or not is_fedmoe_enabled(args):
            return super().get_model_params()

        cfg, client_id, assigned_experts = self._resolve_assignment(args)
        if client_id is None:
            logging.warning("[FedMoE] Cannot determine client_id, upload full state_dict.")
            return state_dict

        filtered_state_dict = filter_state_dict_for_experts(state_dict, assigned_experts)
        removed_count = len(state_dict) - len(filtered_state_dict)
        logging.info(
            "[FedMoE] Client %s upload filtered params. assigned_experts=%s (m=%s/%s), uploaded=%s, removed=%s",
            client_id,
            assigned_experts,
            cfg["experts_per_client"],
            cfg["num_experts"],
            len(filtered_state_dict),
            removed_count,
        )
        return filtered_state_dict

    def train(self, train_data, device, args, test_data=None):
        if is_fedmoe_enabled(args):
            cfg, client_id, assigned_experts = self._resolve_assignment(args)
            if client_id is None:
                logging.warning(
                    "[FedMoE] Cannot determine client_id in train(); skip expert freezing."
                )
            else:
                trainable_experts, frozen_experts = set_trainable_experts(
                    self.model, assigned_experts
                )
                logging.info(
                    "[FedMoE] Client %s deterministic assignment=%s (seed=%s, m=%s/%s). trainable_experts=%s, frozen_experts=%s",
                    client_id,
                    assigned_experts,
                    cfg["assignment_seed"],
                    cfg["experts_per_client"],
                    cfg["num_experts"],
                    trainable_experts,
                    frozen_experts,
                )

        return super().train(train_data, device, args, test_data=test_data)
