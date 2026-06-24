from src.envs.environment import Environment
from src.envs.tokenizers.integer import IntegerTokenizer
from src.envs.tokenizers.number_array import NumberArrayTokenizer
from src.envs.tokenizers.symbolic_int import SymbolicIntTokenizer

# 4 homogeneous coordinates per vertex, KS database has up to 36 vertices.
MAX_VERTICES = 36
MAX_1D_DIM = MAX_VERTICES * 4  # 144

# Coordinate range with headroom over the measured [-903, 147] span.
COORD_MIN, COORD_MAX = -1000, 200

# h11 and other CY invariants fit comfortably in [0, 200].
INVARIANT_MIN, INVARIANT_MAX = 0, 200


class CYEnvironment(Environment):
    """
    Environment for Calabi-Yau reflexive-4-polytope tasks.

    problem_tokenizer : NumberArrayTokenizer over SymbolicIntTokenizer(-1000, 200)
        Encodes a flattened N×4 vertex matrix as "V<4N> c0 c1 ..." where each
        coordinate is a single token (e.g. "-3", "0", "147").

    answer_tokenizer : IntegerTokenizer(base)
        Encodes invariants (h11, h12, …) in positional base notation.
    """

    def __init__(self, params, generator=None, query_tokenizer=None):
        coord_tok = SymbolicIntTokenizer(COORD_MIN, COORD_MAX)
        problem_tokenizer = NumberArrayTokenizer(MAX_1D_DIM, "V", 1, coord_tok)
        answer_tokenizer = IntegerTokenizer(params.base)
        super().__init__(
            params,
            problem_tokenizer=problem_tokenizer,
            query_tokenizer=query_tokenizer,
            answer_tokenizer=answer_tokenizer,
            generator=generator,
        )
