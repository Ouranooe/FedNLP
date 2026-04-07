import logging

from .classification_aggregator import ClassificationAggregator
from .fedmoe_utils import (
    aggregate_fedmoe_params,
    get_fedmoe_config,
    is_fedmoe_enabled,
)


class FedMoEAggregator(ClassificationAggregator):
    @staticmethod
    def _build_weighted_locals(model_list, sample_num_list):
        if len(model_list) == 0:
            return []

        if isinstance(model_list[0], tuple):
            return model_list

        if sample_num_list is not None and len(sample_num_list) == len(model_list):
            return [
                (float(sample_num_list[i]), model_list[i]) for i in range(len(model_list))
            ]
        return [(1.0, local_state) for local_state in model_list]

    def aggregate(self, model_list, sample_num_list):
        if not is_fedmoe_enabled(self.args):
            return super().aggregate(model_list, sample_num_list)

        cfg = get_fedmoe_config(self.args)
        w_locals = self._build_weighted_locals(model_list, sample_num_list)
        prev_global_params = self.model.cpu().state_dict()
        aggregated_params, stats = aggregate_fedmoe_params(
            w_locals=w_locals,
            previous_global_params=prev_global_params,
            num_experts=cfg["num_experts"],
            keep_expert_if_no_contributor=cfg["keep_expert_if_no_contributor"],
        )
        logging.info(
            "[FedMoE Aggregator] modular aggregation enabled. shared_param_count=%s, expert_contributors=%s",
            stats["shared_param_count"],
            stats["expert_contributors"],
        )
        return aggregated_params
