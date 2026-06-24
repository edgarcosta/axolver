from src.envs.environment import Environment
from src.envs.ops.arithmetic import OPERATIONS as _arithmetic
from src.envs.ops.cy_polytope import OPERATIONS as _cy_polytope
from src.envs.ops.graph import OPERATIONS as _graph
from src.envs.ops.integration import OPERATIONS as _integration
from src.envs.ops.matrix import OPERATIONS as _matrix
from src.envs.ops.polynomial import OPERATIONS as _polynomial
from src.envs.ops.synthetic import OPERATIONS as _synthetic

REGISTRY = {}
REGISTRY.update(_arithmetic)
REGISTRY.update(_cy_polytope)
REGISTRY.update(_graph)
REGISTRY.update(_integration)
REGISTRY.update(_matrix)
REGISTRY.update(_polynomial)
REGISTRY.update(_synthetic)


def build_env(params):
    task = params.task
    if task not in REGISTRY:
        raise ValueError(f"Unknown task: {task}")
    built = REGISTRY[task]["build"](params)
    problem_tokenizer = built["problem_tokenizer"]
    query_tokenizer = None
    if "query_tokenizer" in built:
        query_tokenizer = built["query_tokenizer"]
    # Multi-target tasks supply "answer_tokenizers" (a length-K list); single-target
    # tasks supply "answer_tokenizer". Normalize to a list for Environment.
    if "answer_tokenizers" in built:
        answer_tokenizers = built["answer_tokenizers"]
    else:
        answer_tokenizers = [built["answer_tokenizer"]]
    target_names = built.get("target_names")
    generator = built["generator"]

    # Guard rail: the multi-decoder scaffold is only wired for the transformer
    # encoder_decoder. Single-target (K == 1) keeps supporting every architecture.
    if len(answer_tokenizers) > 1:
        assert params.model_type == "transformer", f"n_targets={len(answer_tokenizers)} requires model_type='transformer', got '{params.model_type}'"
        assert (
            params.architecture == "encoder_decoder"
        ), f"n_targets={len(answer_tokenizers)} requires architecture='encoder_decoder', got '{params.architecture}'"
    return Environment(
        params,
        problem_tokenizer=problem_tokenizer,
        query_tokenizer=query_tokenizer,
        answer_tokenizers=answer_tokenizers,
        generator=generator,
        target_names=target_names,
    )
