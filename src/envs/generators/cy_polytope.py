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


class CYPolytopeMultiGenerator(CYPolytopeGenerator):
    """
    Multi-target variant: predict K invariants of a reflexive 4-polytope at once
    (one decoder per invariant). Training is file-backed, so generate() is a
    well-typed no-op stand-in returning (v x 4 int matrix, None, [K ints]).

    evaluate() is exact integer match per target. The evaluator passes the
    single ground-truth int for the relevant target as `answer`, and target_idx
    identifies which invariant is being scored (it does not change the
    comparison, which is a plain exact match).
    """

    def __init__(self, params):
        super().__init__(params)
        self.n_targets = len(params.invariants)

    def generate(self, rng, is_train):
        # No-op stand-in: file-backed training never calls this, but the base
        # class marks generate() abstract, so it must exist and not raise.
        # The K answers are placeholder (all identical): on-the-fly generation is
        # not meaningful here, since training/eval always reload from data files.
        v = rng.integers(5, self.max_vertices + 1)
        m = integer_matrix(self.maxint, v, 4, rng)
        answers = [int(np.abs(m).sum())] * self.n_targets
        return m, None, answers

    def evaluate(self, problem, question, answer, hyp, metrics, target_idx=0):
        if hyp is None:
            return {"is_valid": -1}
        try:
            if int(hyp) == int(answer):
                return {"is_valid": 1}
            return {"is_valid": 0}
        except (TypeError, ValueError):
            return {"is_valid": 0}
