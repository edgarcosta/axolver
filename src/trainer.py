import contextlib
import os
import sys
import time
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.envs.environment import create_train_iterator
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
                logger.info("Encoder parameters frozen.")

        # parse reload_data: supports both "task:path" and "task1:path1;task2:path2"
        self.data_path = None
        self.aux_data_paths = {}
        if params.reload_data != "":
            for pair in params.reload_data.split(";"):
                pair = pair.strip()
                task_name, path = pair.split(":", 1)
                assert os.path.isfile(path), f"Data file not found: {path}"
                if task_name == self.task:
                    self.data_path = path
                else:
                    self.aux_data_paths[task_name] = path

        # open export files if needed
        self.export_files = {}
        if params.export_data and params.is_master:
            export_path = os.path.join(params.dump_path, f"{self.task}.data.prefix")
            self.export_files[self.task] = open(export_path, "w")

        # validate that every decoder_task has a data path
        decoder_tasks = [t.strip() for t in params.decoder_tasks.split(",") if t.strip()] if getattr(params, "decoder_tasks", "") else []
        missing_paths = [t for t in decoder_tasks if t not in self.aux_data_paths]
        if missing_paths:
            raise ValueError(f"--decoder_tasks specifies {missing_paths} but no data path found in --reload_data for those tasks")

        # create data loader(s)
        self.aux_dataloaders = {}
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = iter(create_train_iterator(self.env, self.task, self.data_path, params))
            for aux_task, aux_path in self.aux_data_paths.items():
                self.aux_dataloaders[aux_task] = iter(create_train_iterator(self.env, aux_task, aux_path, params))

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

    def _decoder_labels(self):
        """Return ordered list of decoder labels: [task, aux_task1, aux_task2, ...]."""
        decoder_tasks = [t.strip() for t in self.params.decoder_tasks.split(",") if t.strip()] if getattr(self.params, "decoder_tasks", "") else []
        return [self.task] + decoder_tasks

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

        # Encoder
        enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
        enc_path = os.path.join(self.params.dump_path, f"{name}-encoder.pth")
        logger.info(f"Saving encoder to {enc_path}")
        torch.save({"model": enc_sd, **meta}, enc_path)

        # One file per decoder
        labels = self._decoder_labels()
        for label in labels:
            prefix = "decoder." if label == self.task else f"aux_decoders.{label}."
            dec_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
            dec_path = os.path.join(self.params.dump_path, f"{name}-decoder-{label}.pth")
            logger.info(f"Saving decoder '{label}' to {dec_path}")
            torch.save({"model": dec_sd, **meta}, dec_path)

    def _load_sd_filtered(self, model, ckpt_sd, prefix=""):
        """Load ckpt_sd into model, skipping shape mismatches. Logs what was skipped/missing."""
        current_sd = model.state_dict()
        to_load = {k: v for k, v in ckpt_sd.items() if k in current_sd and v.shape == current_sd[k].shape}
        skipped = [f"{prefix}{k}" for k in ckpt_sd if k not in to_load]
        missing = [f"{prefix}{k}" for k in current_sd if k not in ckpt_sd]
        model.load_state_dict(to_load, strict=False)
        if skipped:
            logger.warning(f"Checkpoint keys skipped (shape mismatch or not in model): {skipped}")
        if missing:
            logger.warning(f"Keys not in checkpoint (randomly initialized): {missing}")

    def reload_checkpoint(self):
        dump = self.params.dump_path
        auto_enc = os.path.join(dump, "checkpoint-encoder.pth")
        explicit = self.params.reload_checkpoint

        if os.path.isfile(auto_enc):
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
                dec_module = model.decoder if label == self.task else model.aux_decoders.get(label)
                if dec_module is None:
                    logger.warning(f"No decoder module for label '{label}', skipping.")
                    continue
                self._load_sd_filtered(dec_module, dec_data["model"], prefix=f"decoder[{label}].")
            data = enc_data  # use encoder file for training state
        elif explicit != "":
            # Single-file layout (old checkpoints or cross-experiment reload)
            assert os.path.isfile(explicit), f"Checkpoint not found: {explicit}"
            logger.info(f"Reloading checkpoint from {explicit} ...")
            data = torch.load(explicit, map_location="cpu", weights_only=False)
            model = _unwrap_model(self.model)
            current_sd = model.state_dict()
            ckpt_sd = data["model"]
            to_load = {k: v for k, v in ckpt_sd.items() if k in current_sd and v.shape == current_sd[k].shape}
            skipped = [k for k in ckpt_sd if k not in to_load]
            missing = [k for k in current_sd if k not in to_load]
            model.load_state_dict(to_load, strict=False)
            if skipped:
                logger.warning(f"Checkpoint keys skipped (shape mismatch or not in model): {skipped}")
            if missing:
                logger.warning(f"Keys not in checkpoint (randomly initialized): {missing}")
        else:
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

    def _fetch_aux_batch_to_device(self, aux_task, device):
        loader = self.aux_dataloaders[aux_task]
        try:
            batch = next(loader)
        except StopIteration:
            self.aux_dataloaders[aux_task] = iter(create_train_iterator(self.env, aux_task, self.aux_data_paths[aux_task], self.params))
            batch = next(self.aux_dataloaders[aux_task])
        _, (dec_tgt, dec_tgt_len), prefix_len = batch
        non_blocking = device in ("cuda", "xpu")
        return (
            dec_tgt.to(device, non_blocking=non_blocking),
            dec_tgt_len.to(device, non_blocking=non_blocking),
            prefix_len.to(device, non_blocking=non_blocking),
        )

    def enc_dec_step(self):
        params = self.params
        device = params.device
        decoder_tasks = [t.strip() for t in params.decoder_tasks.split(",") if t.strip()] if getattr(params, "decoder_tasks", "") else []

        (enc_src, enc_src_len), (dec_tgt, dec_tgt_len), prefix_len = self._fetch_batch_to_device(device)
        bs = dec_tgt_len.size(0)

        if params.architecture == "decoder_only":
            n_tokens = (dec_tgt_len - 1).sum().item()
        elif params.architecture == "encoder_only":
            n_tokens = (enc_src_len - 1).sum().item()
        else:
            n_tokens = (enc_src_len + dec_tgt_len - 2).sum().item()

        if not decoder_tasks:
            # Single-decoder path (original behaviour)
            with self.ctx:
                _, loss = self.model(enc_src, enc_src_len, dec_tgt, dec_tgt_len, prefix_len=prefix_len, task=self.task)
            self.stats["loss"].append(loss.item())
            self.optimize(loss)
        else:
            # Multi-decoder path: encoder runs once, all decoder heads share its output
            dec_tgt_dict = {self.task: dec_tgt}
            dec_tgt_len_dict = {self.task: dec_tgt_len}
            prefix_len_dict = {self.task: prefix_len}
            for aux_task in decoder_tasks:
                aux_dec_tgt, aux_dec_tgt_len, aux_prefix_len = self._fetch_aux_batch_to_device(aux_task, device)
                dec_tgt_dict[aux_task] = aux_dec_tgt
                dec_tgt_len_dict[aux_task] = aux_dec_tgt_len
                prefix_len_dict[aux_task] = aux_prefix_len

            # Build per-task loss weights
            all_tasks = [self.task] + decoder_tasks
            weights_str = getattr(params, "decoder_loss_weights", "")
            if weights_str:
                raw_w = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
                raw_w = (raw_w + [1.0] * len(all_tasks))[: len(all_tasks)]
            else:
                raw_w = [1.0] * len(all_tasks)
            weight_map = dict(zip(all_tasks, raw_w))

            with self.ctx:
                losses = _unwrap_model(self.model).multi_forward(enc_src, enc_src_len, dec_tgt_dict, dec_tgt_len_dict, prefix_len_dict)

            total_loss = sum(weight_map[t] * l for t, l in losses.items())
            self.stats["loss"].append(total_loss.item())
            for t, l in losses.items():
                key = f"loss_{t}"
                if key not in self.stats:
                    self.stats[key] = []
                self.stats[key].append(l.item())
            self.optimize(total_loss)

        self.n_equations += bs
        self.stats["processed_sequences"] += bs
        self.stats["processed_tokens"] += n_tokens
