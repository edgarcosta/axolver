from src.envs.generators.base import Generator


class CYGenerator(Generator):
    """
    Placeholder generator for CY polytope tasks.

    On-the-fly generation is not yet implemented; use --reload_data to load
    pre-computed files produced by scripts/export_axolver_data.py.
    """

    def generate(self, rng, is_train):
        raise NotImplementedError("CY data must be loaded from file via --reload_data")

    def evaluate(self, problem, question, answer, hyp, metrics):
        if hyp is None:
            return {"is_valid": -1}
        return {"is_valid": int(hyp == answer)}

    def encode_class_id(self, problem_data, question_data, answer_data):
        return int(answer_data) if answer_data is not None else 0


def build_cy_polytope(params):
    from src.envs.cy_environment import CYEnvironment

    if not params.reload_data:
        raise RuntimeError("CY data must be loaded from file: pass --reload_data cy_polytope:<path>")
    generator = CYGenerator()
    return CYEnvironment(params, generator=generator)


def register_args(parser):
    parser.add_argument("--base", type=int, default=100,
                        help="Base for IntegerTokenizer used to encode invariants")


OPERATIONS = {
    "cy_polytope": {"build": build_cy_polytope, "register_args": register_args},
}
