import numpy as np

from src.envs.generators.base import Generator
from src.envs.generators.utils import integer_matrix


class CYPolytopeGenerator(Generator):
    """
    Predict an invariant of a Kreuzer-Skarke reflexive 4-polytope from its
    vertex matrix. Training is file-backed (vertex matrix in, invariant out),
    so generate() is never used; it exists only to satisfy the abstract base
    and returns a well-typed (v x 4 int matrix, None, int) stand-in.
    """

    def __init__(self, params):
        self.max_vertices = params.max_vertices
        self.maxint = params.maxint

    def generate(self, rng, is_train):
        # No-op stand-in: file-backed training never calls this, but the base
        # class marks generate() abstract, so it must exist and not raise.
        v = rng.integers(5, self.max_vertices + 1)
        m = integer_matrix(self.maxint, v, 4, rng)
        return m, None, int(np.abs(m).sum())

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        try:
            if int(hyp) == int(answer):
                return {"is_valid": 1}
            return {"is_valid": 0}
        except (TypeError, ValueError):
            return {"is_valid": 0}

    def encode_class_id(self, problem_data, question_data, answer_data):
        # Per-class eval breakdown by vertex count (the matrix has shape (v, 4)).
        return int(problem_data.shape[0])
