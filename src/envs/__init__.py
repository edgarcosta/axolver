from src.envs.environment import Environment
from src.envs.cy_environment import CYEnvironment
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
    if isinstance(built, Environment):
        return built
    problem_tokenizer = built["problem_tokenizer"]
    query_tokenizer = built.get("query_tokenizer")
    answer_tokenizer = built["answer_tokenizer"]
    generator = built["generator"]
    return Environment(
        params, problem_tokenizer=problem_tokenizer, query_tokenizer=query_tokenizer, answer_tokenizer=answer_tokenizer, generator=generator
    )
