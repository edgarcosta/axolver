from src.envs.generators.cy_polytope import CYPolytopeGenerator
from src.envs.tokenizers import IntegerTokenizer, NumberArrayTokenizer


def build_cy_polytope(params):
    return {
        "problem_tokenizer": NumberArrayTokenizer(params.max_vertices, "V", 2, IntegerTokenizer(params.base)),
        "answer_tokenizer": IntegerTokenizer(params.base),
        "generator": CYPolytopeGenerator(params),
    }


def register_args(parser):
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--max_vertices", type=int, default=36, help="Max vertex count (covers V4..V36); do not set below 36")
    parser.add_argument("--maxint", type=int, default=5)


OPERATIONS = {
    "cy_polytope": {"build": build_cy_polytope, "register_args": register_args},
}
