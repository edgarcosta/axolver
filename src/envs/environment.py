from logging import getLogger

from torch.utils.data import DataLoader

from src.dataset import EnvDataset

SPECIAL_WORDS = ["<eos>", "<pad>", "<query>", "<ans>"]

logger = getLogger()


class Environment:
    def __init__(self, params, problem_tokenizer, query_tokenizer, answer_tokenizers, generator, target_names=None):
        self.problem_tokenizer = problem_tokenizer
        self.query_tokenizer = query_tokenizer
        # answer_tokenizers is always a length-K list (K>=1). answer_tokenizer
        # aliases the first one so single-target call sites keep working.
        if not isinstance(answer_tokenizers, (list, tuple)):
            answer_tokenizers = [answer_tokenizers]
        self.answer_tokenizers = list(answer_tokenizers)
        self.answer_tokenizer = self.answer_tokenizers[0]
        self.n_targets = len(self.answer_tokenizers)
        if target_names is None:
            target_names = [None] * self.n_targets
        assert len(target_names) == self.n_targets
        self.target_names = list(target_names)
        params.n_targets = self.n_targets
        self.generator = generator
        self.max_len = params.max_len
        self.max_output_len = params.max_output_len

        all_symbols = set()
        all_symbols.update(self.problem_tokenizer.symbols)
        if self.query_tokenizer is not None:
            all_symbols.update(self.query_tokenizer.symbols)
        for tok in self.answer_tokenizers:
            all_symbols.update(tok.symbols)
        self.words = SPECIAL_WORDS + sorted(all_symbols)
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        self.n_words = params.n_words = len(self.words)
        assert self.word2id["<eos>"] == 0 and self.word2id["<pad>"] == 1 and self.word2id["<query>"] == 2 and self.word2id["<ans>"] == 3
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.query_index = params.query_index = 2
        self.ans_index = params.ans_index = 3

        logger.info(f"words: {self.word2id}")

    def input_to_infix(self, lst):
        return " ".join(lst)

    def output_to_infix(self, lst):
        return " ".join(lst)

    def gen_expr(self, rng, train):
        """
        Generate a (problem, question, answer) triple.
        Returns (problem_tok, question_tok, answer_tok, problem_data, question_data, answer_data, class_id) or None.
        question_tok is [] if no question. question_data is None if no question.
        class_id is the class index from the generator's encode_class_id method (0 during training).

        When n_targets == 1 the answer slot is a single token-list / single object
        (backward compatible). When n_targets > 1 the generator returns a length-K
        list of answers; answer_tok becomes a list of K token-lists and answer_data a
        list of K objects (each encoded with its own tokenizer).
        """
        gen = self.generator.generate(rng, is_train=train)
        if gen is None:
            return None
        problem_data, question_data, answer_data = gen
        problem_tok = self.problem_tokenizer.encode(problem_data)
        question_tok = self.query_tokenizer.encode(question_data) if question_data is not None else []
        enc_len = len(problem_tok) + (1 + len(question_tok) if question_tok else 0)  # +1 for <query>
        if self.max_len > 0 and enc_len >= self.max_len:
            return None

        if self.n_targets == 1:
            answer_tok = self.answer_tokenizer.encode(answer_data)
            if self.max_output_len > 0 and len(answer_tok) >= self.max_output_len:
                return None
        else:
            assert len(answer_data) == self.n_targets, f"generator returned {len(answer_data)} answers, expected {self.n_targets}"
            answer_tok = [tok.encode(a) for tok, a in zip(self.answer_tokenizers, answer_data)]
            if self.max_output_len > 0 and any(len(at) >= self.max_output_len for at in answer_tok):
                return None

        class_id = None if train else self.generator.encode_class_id(problem_data, question_data, answer_data)
        return problem_tok, question_tok, answer_tok, problem_data, question_data, answer_data, class_id

    def check_prediction(self, problem_data, question_data, answer_data, hyp_tokens, metrics, target_idx=0):
        """
        Evaluate a hypothesis against the expected answer for target `target_idx`.
        problem_data, question_data, and answer_data are raw Python objects.
        question_data can be None if no question for this task.
        hyp_tokens is a list of string tokens to be decoded.
        Returns metrics_dict where metrics_dict["is_valid"] is always present.
        """
        hyp_data = self.answer_tokenizers[target_idx].decode(hyp_tokens)
        if self.n_targets == 1:
            # Backward compatible: stock single-target generators take no target_idx.
            metrics_dict = self.generator.evaluate(problem_data, question_data, answer_data, hyp_data, metrics=metrics)
        else:
            metrics_dict = self.generator.evaluate(problem_data, question_data, answer_data, hyp_data, metrics=metrics, target_idx=target_idx)
        assert "is_valid" in metrics_dict
        return metrics_dict


def create_train_iterator(env, task, data_path, params):
    """
    Create a dataset for this environment.
    """
    logger.info(f"Creating train iterator for {task} ...")

    dataset = EnvDataset(env, train=True, params=params, path=data_path)
    num_workers = params.num_workers if data_path is None else 0
    return DataLoader(
        dataset,
        timeout=(0 if num_workers == 0 else 1800),
        batch_size=params.batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        persistent_workers=num_workers > 0,
    )


def create_test_iterator(env, task, data_type, data_path, params):
    """
    Create a dataset for this environment.
    """
    logger.info(f"Creating {data_type} iterator for {task} ...")

    if data_path is None:
        path_iter = None
    elif data_type == "valid":
        path_iter = data_path[0]
    else:
        assert data_type.startswith("test")
        path_iter = data_path[int(data_type[4:])]
    dataset = EnvDataset(env, train=False, params=params, path=path_iter, size=params.eval_size)
    return DataLoader(dataset, timeout=0, batch_size=params.batch_size_eval, num_workers=0, shuffle=False, collate_fn=dataset.collate_fn)
