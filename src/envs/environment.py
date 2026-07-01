from logging import getLogger

from torch.utils.data import DataLoader

from src.dataset import EnvDataset

SPECIAL_WORDS = ["<eos>", "<pad>", "<query>", "<ans>"]

logger = getLogger()


class Environment:
    def __init__(self, params, problem_tokenizer, query_tokenizer, answer_tokenizer, generator):
        self.problem_tokenizer = problem_tokenizer
        self.query_tokenizer = query_tokenizer
        self.answer_tokenizer = answer_tokenizer
        self.generator = generator
        self.max_len = params.max_len
        self.max_output_len = params.max_output_len

        all_symbols = set()
        all_symbols.update(self.problem_tokenizer.symbols)
        if self.query_tokenizer is not None:
            all_symbols.update(self.query_tokenizer.symbols)
        all_symbols.update(self.answer_tokenizer.symbols)
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

        items = list(self.word2id.items())
        excerpt = dict(items[:4] + [("...", "...")] + items[-4:])
        logger.info(f"words ({len(self.word2id)}): {excerpt}")

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
        """
        gen = self.generator.generate(rng, is_train=train)
        if gen is None:
            return None
        problem_data, question_data, answer_data = gen
        problem_tok = self.problem_tokenizer.encode(problem_data)
        question_tok = self.query_tokenizer.encode(question_data) if question_data is not None else []
        answer_tok = self.answer_tokenizer.encode(answer_data)
        enc_len = len(problem_tok) + (1 + len(question_tok) if question_tok else 0)  # +1 for <query>
        if self.max_len > 0 and enc_len >= self.max_len:
            return None
        if self.max_output_len > 0 and len(answer_tok) >= self.max_output_len:
            return None
        class_id = None if train else self.generator.encode_class_id(problem_data, question_data, answer_data)
        return problem_tok, question_tok, answer_tok, problem_data, question_data, answer_data, class_id

    def check_prediction(self, problem_data, question_data, answer_data, hyp_tokens, metrics):
        """
        Evaluate a hypothesis against the expected answer.
        problem_data, question_data, and answer_data are raw Python objects.
        question_data can be None if no question for this task.
        hyp_tokens is a list of string tokens to be decoded.
        Returns metrics_dict where metrics_dict["is_valid"] is always present.
        """
        hyp_data = self.answer_tokenizer.decode(hyp_tokens)
        metrics_dict = self.generator.evaluate(problem_data, question_data, answer_data, hyp_data, metrics=metrics)
        assert "is_valid" in metrics_dict
        return metrics_dict


def parse_column_spec(spec):
    """Parse a column-style data spec into a list of sources.

    Form: 'p,h1,h2:path1;p,h3:path2'  (semicolons separate sources). Within a
    source the comma-separated labels map 1:1 to the file's tab columns: the
    first label is the problem/encoder column, the rest are decoder-head columns.

    Returns a list of {"problem", "heads", "path"} dicts, or None when `spec` is
    not in column form (callers then fall back to legacy "task:path" parsing).
    A source is column-form iff it has a ':' and a ',' before that ':'. This
    distinguishes new specs from legacy reload_data ("task:path", no comma) and
    legacy eval_data ("valid,test", no colon).
    """
    if not spec:
        return None
    segments = [s.strip() for s in spec.split(";") if s.strip()]

    def is_col(seg):
        return ":" in seg and "," in seg.split(":", 1)[0]

    if not any(is_col(seg) for seg in segments):
        return None
    sources = []
    for seg in segments:
        assert is_col(seg), f"cannot mix legacy and column-spec data sources: {seg!r}"
        labels_part, path = seg.split(":", 1)
        labels = [x.strip() for x in labels_part.split(",") if x.strip()]
        assert len(labels) >= 2, f"column source needs a problem label and >=1 head: {seg!r}"
        sources.append({"problem": labels[0], "heads": labels[1:], "path": path.strip()})
    return sources


def heads_from_sources(sources):
    """Ordered union of head labels across sources; the first element is the primary head."""
    heads = []
    for src in sources:
        for h in src["heads"]:
            if h not in heads:
                heads.append(h)
    return heads


def create_train_iterator(env, task, data_path, params, file_heads=None):
    """
    Create a training dataset for this environment.

    file_heads: ordered decoder-head labels matching the file's answer columns
    (columns 1..K; column 0 is the problem). When given, the dataset runs in
    column mode and emits one target per head, all from the same row.
    """
    logger.info(f"Creating train iterator for {task} ...")

    dataset = EnvDataset(env, train=True, params=params, path=data_path, file_heads=file_heads)
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


def create_test_iterator(env, task, data_type, data_path, params, file_heads=None, expose_head=None):
    """
    Create an eval/test dataset for this environment.

    In column mode (file_heads given), data_path holds the single source path and
    expose_head selects which column is scored as the answer for this pass.
    """
    logger.info(f"Creating {data_type} iterator for {task} ...")

    if data_path is None:
        path_iter = None
    elif file_heads is not None:
        path_iter = data_path[0]  # column mode: caller passes the single source path
    elif data_type == "valid":
        path_iter = data_path[0]
    else:
        assert data_type.startswith("test")
        path_iter = data_path[int(data_type[4:])]
    dataset = EnvDataset(
        env, train=False, params=params, path=path_iter, size=params.eval_size, file_heads=file_heads, expose_head=expose_head
    )
    return DataLoader(dataset, timeout=0, batch_size=params.batch_size_eval, num_workers=0, shuffle=False, collate_fn=dataset.collate_fn)
