import os
import sys
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset

logger = getLogger()


class EnvDataset(Dataset):
    def __init__(self, env, train, params, path, size=None):
        super().__init__()
        self.env = env
        self.train = train
        # Number of prediction targets (K). K == 1 is the stock single-target path;
        # K > 1 means one decoder target per invariant (multi-target tasks).
        self.n_targets = getattr(env, "n_targets", 1)
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.two_classes = params.two_classes
        self.first_class_prob = params.first_class_prob
        self.first_class_size = params.first_class_size

        self.decoder_only = params.architecture == "decoder_only"
        self.encoder_only = params.architecture == "encoder_only"
        self.export_data = params.export_data and train
        assert size is None or not self.train

        # batching
        self.num_workers = params.num_workers if path is None else 0

        self.index_dataset = params.index_dataset
        self.max_examples = params.max_examples
        self.reload_size = params.reload_size
        self.batch_load = path is not None and params.reload_size > 0 and not self.index_dataset
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0
        self._fh = None

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            if self.batch_load and self.train:
                self.load_chunk()
            elif self.index_dataset:
                self.offsets = self._build_index()
            else:
                max_lines = self.max_examples if self.max_examples > 0 else None
                filter_by_rank = train
                self.data = self.read_lines(max_lines=max_lines, filter_by_rank=filter_by_rank)
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        if self.two_classes and path is not None and not (self.batch_load and self.train) and not self.index_dataset:
            assert len(self.data) > self.first_class_size

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            n = len(self.offsets) if self.index_dataset else (10000 if path is None else len(self.data))
            self.size = n
        else:
            assert size > 0
            self.size = size

    def _split_parts(self, parts):
        """Split a tab-separated line's parts into (problem, question, answers).

        Supports K = self.n_targets targets. Valid layouts are:
          - 1 + K columns: problem + K answers (no question)
          - 2 + K columns: problem + question + K answers
        `answers` is always a list of length K (length 1 for the stock path).
        Returns None on an invalid column count.
        """
        k = self.n_targets
        n = len(parts)
        if n == 1 + k:
            return parts[0], None, list(parts[1:])
        if n == 2 + k:
            return parts[0], parts[1], list(parts[2:])
        return None

    def _valid_ncols(self, n):
        return n in (1 + self.n_targets, 2 + self.n_targets)

    def _decode_answers(self, answer):
        """Decode a list of K answer token-lists into answer_data.

        Single-target (K == 1) returns the single decoded object (backward
        compatible); multi-target returns a length-K list of decoded objects,
        each parsed with its own tokenizer.
        """
        if self.n_targets == 1:
            return self.env.answer_tokenizers[0].decode(answer[0])
        return [tok.decode(a) for tok, a in zip(self.env.answer_tokenizers, answer)]

    def _build_index(self):
        offsets = []
        with open(self.path, "rb") as f:
            n_read = 0
            while True:
                if self.train and self.max_examples > 0 and n_read >= self.max_examples:
                    break
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                n_read += 1
                if self.train and (n_read - 1) % self.n_gpu_per_node != self.local_rank:
                    continue
                parts = line.rstrip(b"\n").split(b"\t")
                if not self._valid_ncols(len(parts)):
                    continue
                offsets.append(off)
        return np.array(offsets, dtype=np.int64)

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")

    def load_chunk(self):
        self.basepos = self.nextpos
        self.data = self.read_lines(max_lines=self.reload_size, filter_by_rank=True)
        self.nextpos = self.basepos + len(self.data)
        if len(self.data) == 0:
            self.load_chunk()

    def read_lines(self, max_lines=None, filter_by_rank=True):
        logger.info(f"Loading data from {self.path} ... seekpos {self.seekpos}")
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            n_read = 0
            for line in f:
                if max_lines is not None and n_read >= max_lines:
                    break
                n_read += 1
                if not filter_by_rank or (n_read - 1) % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip())
            endfile = max_lines is None or n_read < max_lines
            self.seekpos = 0 if endfile else f.tell()

        # Format: problem [\t question] \t answer_0 [\t answer_1 ...] (K answers).
        # Each stored entry is [problem_str, question_str_or_None, [answer_str, ...]]
        # with the answer list of length K (length 1 for the single-target path).
        data = []
        for line in lines:
            parts = line.split("\t")
            split = self._split_parts(parts)
            if split is None:
                continue
            problem_str, question_str, answer_strs = split
            data.append([problem_str, question_str, answer_strs])
        logger.info(f"Loaded {len(data)} equations from the disk.")
        return data

    def batch_sequences(self, sequences, bos=True, eos=True, left_pad=False):
        pad_index = self.env.pad_index
        eos_index = self.env.eos_index
        lengths = [len(s) + int(bos) + int(eos) for s in sequences]
        max_len = max(lengths)
        sent = np.full((len(sequences), max_len), pad_index, dtype=np.int64)

        for i, s in enumerate(sequences):
            offset = (max_len - lengths[i]) if left_pad else 0
            if bos:
                sent[i, offset] = eos_index
            sent[i, offset + int(bos) : offset + int(bos) + len(s)] = s
            if eos:
                sent[i, offset + lengths[i] - 1] = eos_index

        return torch.from_numpy(sent), torch.tensor(lengths, dtype=torch.long)

    def collate_fn(self, elements):
        if self.train:
            problem, question, answer = zip(*elements)
            if self.export_data:
                # Export is single-target (K == 1); unwrap the per-target list so the
                # exported answer is a flat token list (" ".join in trainer.export_data).
                # Fail loudly rather than silently dropping targets 1..K-1.
                assert self.env.n_targets == 1, "export_data is only supported for single-target tasks"
                return list(problem), list(question), [a[0] for a in answer]
        else:
            problem, question, answer, problem_data, question_data, answer_data, class_id = zip(*elements)

        enc_seqs = []
        for pi, qi in zip(problem, question):
            seq = list(pi)
            if qi:
                seq += ["<query>"] + list(qi)
            enc_seqs.append(seq)

        if self.decoder_only:
            enc_problem, enc_problem_len = None, None
        else:
            enc_problem, enc_problem_len = self.batch_sequences([[self.env.word2id[w] for w in seq] for seq in enc_seqs], bos=False)

        # answer[i] is a list of K token-lists (one per target). Build one decoder
        # target per target index t. The prefix (and thus prefix_len) is the same
        # across targets, so compute it once. K > 1 is encoder_decoder-only (guarded
        # in build_env), where the prefix is just ["<ans>"].
        k = self.n_targets
        dec_targets = []
        prefix_lens = None
        for t in range(k):
            dec_seqs, pls = [], []
            for enc_seq, ai in zip(enc_seqs, answer):
                prefix = enc_seq + ["<ans>"] if self.decoder_only else ["<ans>"]
                dec_seqs.append(prefix + list(ai[t]))
                pls.append(1 + len(prefix))
            dec_tgt_t, dec_tgt_len_t = self.batch_sequences([[self.env.word2id[w] for w in seq] for seq in dec_seqs], bos=not self.encoder_only)
            dec_targets.append((dec_tgt_t, dec_tgt_len_t))
            if prefix_lens is None:
                prefix_lens = pls
        prefix_len = torch.tensor(prefix_lens, dtype=torch.long)

        if self.train:
            return (enc_problem, enc_problem_len), dec_targets, prefix_len

        class_id = list(class_id)

        gen_seqs = []
        for enc_seq in enc_seqs:
            prefix = enc_seq + ["<ans>"] if self.decoder_only else ["<ans>"]
            gen_seqs.append([self.env.word2id[w] for w in prefix])

        gen_prefix, gen_prefix_len = self.batch_sequences(gen_seqs, bos=True, eos=False, left_pad=True)
        ref_answers = [self.batch_sequences([[self.env.word2id[w] for w in ai[t]] for ai in answer], bos=False) for t in range(k)]
        return (
            (enc_problem, enc_problem_len),
            dec_targets,
            prefix_len,
            (gen_prefix, gen_prefix_len),
            ref_answers,
            class_id,
            list(problem_data),
            list(question_data),
            list(answer_data),
        )

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.default_rng([worker_id, self.global_rank, self.env_base_seed])
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed})."
            )
        else:
            self.rng = np.random.default_rng()

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        if self.index_dataset:
            return self._read_sample_indexed(index)

        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                if self.two_classes:
                    if self.rng.random() < self.first_class_prob:
                        idx = self.rng.integers(self.first_class_size) % len(self.data)
                    else:
                        idx = (self.first_class_size + self.rng.integers(len(self.data) - self.first_class_size)) % len(self.data)
                else:
                    idx = self.rng.integers(len(self.data))

        problem_str, question_str, answer_strs = self.data[idx]
        problem = problem_str.split()
        question = question_str.split() if question_str else None
        # answer is a list of K token-lists (length 1 for the single-target path).
        answer = [a.split() for a in answer_strs]
        assert len(problem) >= 1 and all(len(a) >= 1 for a in answer)
        if self.train:
            return problem, question, answer
        problem_data = self.env.problem_tokenizer.decode(problem)
        if self.env.query_tokenizer is not None:
            question_data = self.env.query_tokenizer.decode(question) if question else None
        else:
            question_data = None
        answer_data = self._decode_answers(answer)
        class_id = self.env.generator.encode_class_id(problem_data, question_data, answer_data)
        return problem, question, answer, problem_data, question_data, answer_data, class_id

    def _read_sample_indexed(self, index):
        self._ensure_open()
        n = len(self.offsets)
        j = int(self.rng.integers(n)) if self.train else index % n
        off = int(self.offsets[j])
        self._fh.seek(off)
        line = self._fh.readline().decode("utf-8").rstrip("\n")
        parts = line.split("\t")
        split = self._split_parts(parts)
        if split is None:
            return self._read_sample_indexed(index)
        problem_str, question_str, answer_strs = split
        problem = problem_str.split()
        question = question_str.split() if question_str else None
        answer = [a.split() for a in answer_strs]
        if not problem or any(len(a) == 0 for a in answer):
            return self._read_sample_indexed(index)
        if self.train:
            return problem, question, answer
        problem_data = self.env.problem_tokenizer.decode(problem)
        if self.env.query_tokenizer is not None:
            question_data = self.env.query_tokenizer.decode(question) if question else None
        else:
            question_data = None
        answer_data = self._decode_answers(answer)
        class_id = self.env.generator.encode_class_id(problem_data, question_data, answer_data)
        return problem, question, answer, problem_data, question_data, answer_data, class_id

    def generate_sample(self):
        while True:
            try:
                result = self.env.gen_expr(rng=self.rng, train=self.train)
                if result is None:
                    continue
                problem, question, answer, problem_data, question_data, answer_data, class_id = result
                break
            except Exception as e:
                logger.error(
                    f"An unknown exception of type {type(e).__name__} occurred for worker {self.get_worker_id()} in line {sys.exc_info()[-1].tb_lineno}. Arguments:{e.args!r}."
                )
                continue
        # Normalize the answer to a list of K token-lists so the train collate path
        # is uniform with read_sample. gen_expr already returns a length-K list when
        # n_targets > 1; for the single-target path it returns a flat token list.
        if self.n_targets == 1:
            answer = [answer]
        if self.train:
            return problem, question, answer
        else:
            return problem, question, answer, problem_data, question_data, answer_data, class_id
