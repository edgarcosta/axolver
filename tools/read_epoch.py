#!/usr/bin/env python3
"""Read the epoch count from a PyTorch checkpoint without loading tensors."""
import sys
import zipfile
import pickle


class _Skip:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return _Skip()

    def find_class(self, module, name):
        if module.startswith("torch"):
            return _Skip
        return super().find_class(module, name)


try:
    with zipfile.ZipFile(sys.argv[1]) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith(".pkl"))
        with zf.open(pkl_name) as f:
            data = _Unpickler(f).load()
    print(data.get("epoch", "?"))
except Exception:
    print("?")
