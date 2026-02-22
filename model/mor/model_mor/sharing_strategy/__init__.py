from model_mor.sharing_strategy.llama import sharing_strategy as sharing_strategy_llama
from model_mor.sharing_strategy.gpt_neox import sharing_strategy as sharing_strategy_gpt_neox
from model_mor.sharing_strategy.gemma3 import sharing_strategy as sharing_strategy_gemma3


SHARING_STRATEGY = {
    "tinyllama": sharing_strategy_llama,
    "olmo": sharing_strategy_llama,
    "smollm": sharing_strategy_llama,
    "smollm2": sharing_strategy_llama,
    "pythia": sharing_strategy_gpt_neox,
    "gemma3": sharing_strategy_gemma3,
}