import contextlib
import os
import sys
import time
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.envs.environment import create_train_iterator, parse_column_spec
from src.optim import get_optimizer

logger = getLogger()


def _unwrap_model(model):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def is_bf16_supported(device):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_fp16_supported(device):
    try:
        torch.tensor([1.0], dtype=torch.float16, device=device)
        return True
    except Exception:
        return False


def default_dtype(device, amp):
    if amp:
        if is_bf16_supported(device):
            return torch.bfloat16
        elif is_fp16_supported(device):
            return torch.float16
    return torch.float32


def setup_amp(device, dtype):
    use_amp = dtype in (torch.float16, torch.bfloat16)
    if not use_amp:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device, dtype=dtype, enabled=True)


class Trainer:
    def __init__(self, model, env, params):
        self.model = model
        self.params = params
        self.env = env
        self.task = params.task

        # Decoder heads (from the column spec, resolved into params.decoder_heads by
        # train.py). First head is primary (model.decoder); the rest are aux_decoders.
        # Legacy mode (no column spec) => a single head == the env task. Set before
        # reload_checkpoint(), which uses self.heads / self.primary_head.
        self.heads = list(getattr(params, "decoder_heads", []) or [self.task])
        self.primary_head = self.heads[0]
        self.head_weights = self._parse_head_weights()

        assert params.report_loss_every > 0

        # distributed training
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True
            )

        self.optimizer, self.scheduler = get_optimizer(_unwrap_model(self.model).parameters(), params.optimizer)

        self.dtype = default_dtype(params.device, params.amp)
        self.ctx = setup_amp(params.device, self.dtype)
        self.scaler = torch.amp.GradScaler(device=params.device, enabled=(self.dtype == torch.float16))

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            # Metrics prefixed with "_" are minimized (e.g. "_loss"), others are maximized.
            if m.startswith("_"):
                self.metrics.append((m[1:], False))
            else:
                self.metrics.append((m, True))
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        self.epoch_size = params.epoch_size

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_equations = 0
        self.stats = {"processed_sequences": 0, "processed_tokens": 0, "loss": []}
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # freeze encoder if requested (applied after checkpoint reload)
        if getattr(params, "freeze_encoder", False):
            enc = getattr(_unwrap_model(self.model), "encoder", None)
            if enc is not None:
                enc.requires_grad_(False)
                enc.eval()  # disable dropout so the frozen features are deterministic
                self._frozen_encoder = enc
                logger.info("Encoder parameters frozen.")

        # Column-mode training sources (one source = aligned multi-head; many = alternate).
        self.sources = parse_column_spec(params.reload_data) if params.reload_data else None

        # Legacy single-decoder data path: first "task:path" segment.
        self.data_path = None
        if self.sources is None and params.reload_data != "":
            seg = params.reload_data.split(";")[0].strip()
            _, path = seg.split(":", 1)
            assert os.path.isfile(path), f"Data file not found: {path}"
            self.data_path = path

        # open export files if needed
        self.export_files = {}
        if params.export_data and params.is_master:
            export_path = os.path.join(params.dump_path, f"{self.task}.data.prefix")
            self.export_files[self.task] = open(export_path, "w")

        # create data loader(s)
        self.dataloaders = []
        self._src_idx = 0
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            if self.sources is not None:
                # One loader per source. A single source trains all its heads from one
                # encoder pass (aligned); multiple sources alternate per batch (no
                # cross-source row alignment required).
                for src in self.sources:
                    for path in [src["path"]]:
                        assert os.path.isfile(path), f"Data file not found: {path}"
                    self.dataloaders.append(
                        iter(create_train_iterator(self.env, self.task, src["path"], params, file_heads=src["heads"]))
                    )
                logger.info(f"Column-mode training: heads={self.heads}, sources={len(self.dataloaders)} "
                            f"({'aligned' if len(self.dataloaders) == 1 else 'alternating'})")
            else:
                self.dataloader = iter(create_train_iterator(self.env, self.task, self.data_path, params))

    def _parse_head_weights(self):
        """Map each head to its loss weight from --decoder_loss_weights (positional, [primary, aux1, ...])."""
        weights_str = getattr(self.params, "decoder_loss_weights", "")
        if not weights_str:
            return {h: 1.0 for h in self.heads}
        raw = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
        raw = (raw + [1.0] * len(self.heads))[: len(self.heads)]
        return dict(zip(self.heads, raw))

    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.info("NaN detected")
            self.save_checkpoint("checkpoint_nan")
            exit()

        self.scaler.scale(loss).backward()

        if self.params.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(_unwrap_model(self.model).parameters(), self.params.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    def iter(self):
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.params.report_loss_every == 0:
            self.print_stats()

    def print_stats(self):
        s_iter = f"{self.n_total_iter:7d} - "
        s_stat = " || ".join([f"{k.upper().replace('_', '-')}: {np.mean(v):7.4f}" for k, v in self.stats.items() if type(v) is list and len(v) > 0])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rate
        s_lr = " - LR: " + " / ".join(f"{group['lr']:.4e}" for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = f"{self.stats['processed_sequences'] * 1.0 / diff:7.2f} examples/s - {self.stats['processed_tokens'] * 1.0 / diff:8.2f} words/s - "
        self.stats["processed_sequences"] = 0
        self.stats["processed_tokens"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def reset_epoch_stats(self):
        self.n_equations = 0
        self.model.train()
        # keep a frozen encoder in eval mode (model.train() would re-enable dropout)
        if getattr(self, "_frozen_encoder", None) is not None:
            self._frozen_encoder.eval()

    def _decoder_labels(self):
        """Ordered list of decoder head labels (primary first, then aux heads)."""
        return list(self.heads)

    def save_checkpoint(self, name):
        if not self.params.is_master:
            return

        model = _unwrap_model(self.model)
        full_sd = model.state_dict()

        meta = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "decrease_counts": getattr(self, "decrease_counts", 0),
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        # Encoder (skip if frozen -- weights are unchanged)
        if not self.params.freeze_encoder:
            enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
            enc_path = os.path.join(self.params.dump_path, f"{name}-encoder.pth")
            logger.info(f"Saving encoder to {enc_path}")
            torch.save({"model": enc_sd, **meta}, enc_path)

        # One file per decoder
        labels = self._decoder_labels()
        for label in labels:
            prefix = "decoder." if label == self.primary_head else f"aux_decoders.{label}."
            dec_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
            dec_path = os.path.join(self.params.dump_path, f"{name}-decoder-{label}.pth")
            logger.info(f"Saving decoder '{label}' to {dec_path}")
            torch.save({"model": dec_sd, **meta}, dec_path)

    def _load_sd_filtered(self, model, ckpt_sd, prefix="", must_be_complete=False):
        """Load ckpt_sd into model, skipping shape mismatches. Logs what was skipped/missing.

        If must_be_complete is set, raise when anything fails to load -- used for a
        frozen encoder, where a silently-skipped tensor (e.g. a vocab-size-mismatched
        token embedding) would be frozen at random init and quietly cripple the model.
        """
        current_sd = model.state_dict()
        to_load = {k: v for k, v in ckpt_sd.items() if k in current_sd and v.shape == current_sd[k].shape}
        skipped = [f"{prefix}{k}" for k in ckpt_sd if k not in to_load]
        missing = [f"{prefix}{k}" for k in current_sd if k not in ckpt_sd]
        if must_be_complete and (skipped or missing):
            raise ValueError(
                f"Incomplete load into a frozen module: skipped={skipped}, missing={missing}. "
                f"This usually means a vocabulary/shape mismatch between checkpoints; "
                f"freezing would lock in randomly-initialized weights."
            )
        model.load_state_dict(to_load, strict=False)
        if skipped:
            logger.warning(f"Checkpoint keys skipped (shape mismatch or not in model): {skipped}")
        if missing:
            logger.warning(f"Keys not in checkpoint (randomly initialized): {missing}")

    def reload_checkpoint(self):
        dump = self.params.dump_path
        auto_enc = os.path.join(dump, "checkpoint-encoder.pth")
        explicit = self.params.reload_checkpoint
        explicit_enc = getattr(self.params, "reload_encoder_checkpoint", "")
        explicit_dec = getattr(self.params, "reload_decoder_checkpoint", "")

        # Only a genuine same-experiment auto-resume should adopt the donor's
        # training clock (epoch / optimizer / scheduler / best-metrics). Any
        # partial or cross-experiment warm-start starts from a fresh clock.
        resume_state = False
        freeze = getattr(self.params, "freeze_encoder", False)

        # A local checkpoint in dump_path means this is a same-experiment resume
        # (e.g. a SLURM requeue). It must take precedence over warm-start reload
        # flags, otherwise the requeued job would re-warm-start and restart the
        # decoder from epoch 0 instead of continuing where it left off.
        if os.path.isfile(auto_enc) and (explicit_enc or explicit_dec or explicit):
            logger.info("Local checkpoint found in dump_path; ignoring reload flags and resuming.")
            explicit_enc = explicit_dec = explicit = ""

        if explicit_enc != "" or explicit_dec != "":
            # Mixed-source: load encoder and decoder from separate explicit paths
            model = _unwrap_model(self.model)
            data = None
            if explicit_enc != "":
                assert os.path.isfile(explicit_enc), f"Encoder checkpoint not found: {explicit_enc}"
                logger.info(f"Reloading encoder from {explicit_enc} ...")
                enc_data = torch.load(explicit_enc, map_location="cpu", weights_only=False)
                self._load_sd_filtered(model.encoder, enc_data["model"], prefix="encoder.", must_be_complete=freeze)
                data = enc_data
            if explicit_dec != "":
                assert os.path.isfile(explicit_dec), f"Decoder checkpoint not found: {explicit_dec}"
                logger.info(f"Reloading decoder from {explicit_dec} ...")
                dec_data = torch.load(explicit_dec, map_location="cpu", weights_only=False)
                self._load_sd_filtered(model.decoder, dec_data["model"], prefix=f"decoder[{self.task}].")
                data = dec_data  # prefer decoder state: its epoch is always current
            if data is None:
                return
            # Decoder checkpoint is authoritative for training state: it always
            # reflects the current run's epoch, optimizer, and scheduler, even
            # when the encoder comes from a different (e.g. frozen) experiment.
            if explicit_dec:
                resume_state = True
        elif os.path.isfile(auto_enc):
            # Multifile layout: load encoder + each decoder from separate files
            logger.info(f"Reloading multifile checkpoint from {dump} ...")
            model = _unwrap_model(self.model)
            enc_data = torch.load(auto_enc, map_location="cpu", weights_only=False)
            self._load_sd_filtered(model.encoder, enc_data["model"], prefix="encoder.")
            for label in self._decoder_labels():
                dec_path = os.path.join(dump, f"checkpoint-decoder-{label}.pth")
                if not os.path.isfile(dec_path):
                    logger.warning(f"Decoder checkpoint not found, skipping: {dec_path}")
                    continue
                dec_data = torch.load(dec_path, map_location="cpu", weights_only=False)
                aux = getattr(model, "aux_decoders", {})
                dec_module = model.decoder if label == self.primary_head else (aux[label] if label in aux else None)
                if dec_module is None:
                    logger.warning(f"No decoder module for label '{label}', skipping.")
                    continue
                self._load_sd_filtered(dec_module, dec_data["model"], prefix=f"decoder[{label}].")
            data = enc_data  # use encoder file for training state
            resume_state = True  # genuine same-experiment auto-resume
        elif explicit != "":
            # Explicit single checkpoint: a full-model file, or an encoder/decoder
            # shard saved with submodule-local (prefix-stripped) keys.
            assert os.path.isfile(explicit), f"Checkpoint not found: {explicit}"
            model = _unwrap_model(self.model)
            base = os.path.basename(explicit)
            data = torch.load(explicit, map_location="cpu", weights_only=False)
            if base.endswith("-encoder.pth"):
                logger.info(f"Reloading encoder from {explicit} ...")
                self._load_sd_filtered(model.encoder, data["model"], prefix="encoder.", must_be_complete=freeze)
            elif "-decoder-" in base:
                logger.info(f"Reloading decoder from {explicit} ...")
                self._load_sd_filtered(model.decoder, data["model"], prefix=f"decoder[{self.task}].")
            else:
                logger.info(f"Reloading checkpoint from {explicit} ...")
                current_sd = model.state_dict()
                ckpt_sd = data["model"]
                to_load = {k: v for k, v in ckpt_sd.items() if k in current_sd and v.shape == current_sd[k].shape}
                if not to_load:
                    raise ValueError(
                        f"No parameters from {explicit} matched the model. This usually means an "
                        f"encoder/decoder shard was passed to --reload_checkpoint; use "
                        f"--reload_encoder_checkpoint / --reload_decoder_checkpoint instead."
                    )
                skipped = [k for k in ckpt_sd if k not in to_load]
                missing = [k for k in current_sd if k not in to_load]
                if freeze:
                    enc_bad = [k for k in skipped + missing if k.startswith("encoder.")]
                    if enc_bad:
                        raise ValueError(
                            f"freeze_encoder is set but these encoder tensors did not load from "
                            f"{explicit} (would be frozen at random init): {enc_bad}. Likely a "
                            f"vocabulary/shape mismatch between checkpoints."
                        )
                model.load_state_dict(to_load, strict=False)
                if skipped:
                    logger.warning(f"Checkpoint keys skipped (shape mismatch or not in model): {skipped}")
                if missing:
                    logger.warning(f"Keys not in checkpoint (randomly initialized): {missing}")
        else:
            return

        if not resume_state:
            # Partial / cross-experiment warm-start: load weights only and keep a
            # fresh training clock so the new run isn't polluted by the donor's
            # epoch, optimizer moments, scheduler, or best-metric stopping state.
            logger.info("Loaded weights only (fresh optimizer / epoch / best-metrics).")
            return

        try:
            self.optimizer.load_state_dict(data["optimizer"])
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Could not load optimizer state (starting fresh): {e}")
        if self.scheduler is not None and data.get("scheduler") is not None:
            self.scheduler.load_state_dict(data["scheduler"])
        if "scaler" in data:
            self.scaler.load_state_dict(data["scaler"])
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        if hasattr(self, "decrease_counts"):
            self.decrease_counts = data.get("decrease_counts", 0)
        logger.info(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_best_model(self, scores):
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.info(f'Metric "{metric}" not found in scores!')
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info(f"New best score for {metric}: {scores[metric]:.6f}")
                self.save_checkpoint(f"best-{metric}")

    def end_epoch(self, scores):
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and self.params.is_master:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(f"New best validation score: {self.best_stopping_criterion}")
                self.decrease_counts = 0
            else:
                logger.info(f"Not a better validation score ({self.decrease_counts} / {self.decrease_counts_max}).")
                self.decrease_counts += 1
            if self.decrease_counts >= self.decrease_counts_max:
                logger.info(
                    f"Stopping criterion has been below its best value for more " f"than {self.decrease_counts_max} epochs. Ending the experiment..."
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        try:
            batch = next(self.dataloader)
        except Exception as e:
            logger.error(
                f"An unknown exception of type {type(e).__name__} occurred in line {sys.exc_info()[-1].tb_lineno} when fetching batch. Arguments:{e.args!r}. Restarting ..."
            )
            if self.params.is_master and "SLURM_JOB_ID" in os.environ:
                logger.info(f"Requeuing job {os.environ['SLURM_JOB_ID']}")
                os.system(f"scontrol requeue {os.environ['SLURM_JOB_ID']}")
            raise

        return batch

    def export_data(self):
        problem_list, question_list, answer_list = self.get_batch()
        for problem_tok, question_tok, answer_tok in zip(problem_list, question_list, answer_list):
            if len(problem_tok) == 0 or len(answer_tok) == 0:
                continue
            f = self.export_files[self.task]
            if question_tok:
                f.write(" ".join(problem_tok) + "\t" + " ".join(question_tok) + "\t" + " ".join(answer_tok) + "\n")
            else:
                f.write(" ".join(problem_tok) + "\t" + " ".join(answer_tok) + "\n")
        self.n_equations += len(problem_list)

    def close_export_files(self):
        for f in self.export_files.values():
            f.close()
        self.export_files = {}

    def _fetch_batch_to_device(self, device):
        (enc_src, enc_src_len), (dec_tgt, dec_tgt_len), prefix_len = self.get_batch()
        non_blocking = device in ("cuda", "xpu")
        if enc_src is not None:
            enc_src = enc_src.to(device, non_blocking=non_blocking)
        if enc_src_len is not None:
            enc_src_len = enc_src_len.to(device, non_blocking=non_blocking)
        dec_tgt = dec_tgt.to(device, non_blocking=non_blocking)
        dec_tgt_len = dec_tgt_len.to(device, non_blocking=non_blocking)
        prefix_len = prefix_len.to(device, non_blocking=non_blocking)
        return (enc_src, enc_src_len), (dec_tgt, dec_tgt_len), prefix_len

    def enc_dec_step(self):
        if self.sources is not None:
            return self._column_step()

        params = self.params
        device = params.device
        (enc_src, enc_src_len), (dec_tgt, dec_tgt_len), prefix_len = self._fetch_batch_to_device(device)
        bs = dec_tgt_len.size(0)

        if params.architecture == "decoder_only":
            n_tokens = (dec_tgt_len - 1).sum().item()
        elif params.architecture == "encoder_only":
            n_tokens = (enc_src_len - 1).sum().item()
        else:
            n_tokens = (enc_src_len + dec_tgt_len - 2).sum().item()

        with self.ctx:
            _, loss = self.model(enc_src, enc_src_len, dec_tgt, dec_tgt_len, prefix_len=prefix_len, task=self.primary_head)
        self.stats["loss"].append(loss.item())
        self.optimize(loss)

        self.n_equations += bs
        self.stats["processed_sequences"] += bs
        self.stats["processed_tokens"] += n_tokens

    def _column_step(self):
        """One training step in column mode. A single source trains all its heads from one
        encoder pass (aligned); with multiple sources we round-robin per batch (alternating)."""
        device = self.params.device
        non_blocking = device in ("cuda", "xpu")

        loader = self.dataloaders[self._src_idx]
        self._src_idx = (self._src_idx + 1) % len(self.dataloaders)
        (enc_src, enc_src_len), head_targets = next(loader)

        enc_src = enc_src.to(device, non_blocking=non_blocking)
        enc_src_len = enc_src_len.to(device, non_blocking=non_blocking)
        dec_tgt_dict, dec_tgt_len_dict, prefix_len_dict = {}, {}, {}
        for h, (d, dl, pl) in head_targets.items():
            dec_tgt_dict[h] = d.to(device, non_blocking=non_blocking)
            dec_tgt_len_dict[h] = dl.to(device, non_blocking=non_blocking)
            prefix_len_dict[h] = pl.to(device, non_blocking=non_blocking)

        bs = enc_src_len.size(0)
        any_len = next(iter(dec_tgt_len_dict.values()))
        n_tokens = (enc_src_len + any_len - 2).sum().item()

        with self.ctx:
            losses = _unwrap_model(self.model).multi_forward(enc_src, enc_src_len, dec_tgt_dict, dec_tgt_len_dict, prefix_len_dict)

        total_loss = sum(self.head_weights.get(h, 1.0) * l for h, l in losses.items())
        self.stats["loss"].append(total_loss.item())
        for h, l in losses.items():
            self.stats.setdefault(f"loss_{h}", []).append(l.item())
        self.optimize(total_loss)

        self.n_equations += bs
        self.stats["processed_sequences"] += bs
        self.stats["processed_tokens"] += n_tokens
