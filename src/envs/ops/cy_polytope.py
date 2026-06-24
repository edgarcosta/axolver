from src.envs.generators.cy_polytope import CYPolytopeGenerator, CYPolytopeMultiGenerator
from src.envs.tokenizers import IntegerTokenizer, NumberArrayTokenizer

# Default invariants, in the exact column order of data/axolver_b10 (meta.json).
# This order is load-bearing: it MUST match the data columns (problem + these K).
DEFAULT_INVARIANTS = [
    "h11",
    "h12",
    "point_count",
    "dual_point_count",
    "facet_count",
    "euler_characteristic",
    "vertex_count",
]


def _invariant_list(value):
    return [s for s in value.split(",") if s]


def build_cy_polytope(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(params.max_vertices, "V", 2, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": CYPolytopeGenerator(params),
    }


def build_cy_polytope_multi(params):
    # Normalize --invariants to a list of names (argparse may hand us a raw string).
    if isinstance(params.invariants, str):
        params.invariants = _invariant_list(params.invariants)
    assert len(params.invariants) >= 1, "cy_polytope_multi requires at least one invariant"
    k = len(params.invariants)
    return {
        "problem_tokenizer": NumberArrayTokenizer(params.max_vertices, "V", 2, IntegerTokenizer(params.base)),
        # One answer tokenizer per target; all integers in the shared base.
        "answer_tokenizers": [IntegerTokenizer(params.base) for _ in range(k)],
        "target_names": list(params.invariants),
        "generator": CYPolytopeMultiGenerator(params),
    }


def register_args(parser):
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--max_vertices", type=int, default=36, help="Max vertex count (covers V4..V36); do not set below 36")
    parser.add_argument("--maxint", type=int, default=5)


def register_args_multi(parser):
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--max_vertices", type=int, default=36, help="Max vertex count (covers V4..V36); do not set below 36")
    parser.add_argument("--maxint", type=int, default=5)
    parser.add_argument(
        "--invariants",
        type=_invariant_list,
        default=list(DEFAULT_INVARIANTS),
        help=(
            "Comma-separated invariant names, one decoder per name. Order MUST match "
            "the data columns (problem + these K). Default: " + ",".join(DEFAULT_INVARIANTS)
        ),
    )


OPERATIONS = {
    "cy_polytope": {"build": build_cy_polytope, "register_args": register_args},
    "cy_polytope_multi": {"build": build_cy_polytope_multi, "register_args": register_args_multi},
}
