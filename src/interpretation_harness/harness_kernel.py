"""Alias for kernel.py.

run_interp_harness.py loads its comparison engine from a sibling named
harness_kernel.py; the test suite loads the same engine from kernel.py. Rather
than ship two byte-identical 47 KB copies, this module re-exports kernel.py so
there is a single source of truth. Loaded by file path via importlib, exactly
like the original copy was.
"""
import importlib.util as _u
import os as _os

_kernel_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernel.py")
_spec = _u.spec_from_file_location("cvs_cmp_kernel", _kernel_path)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export every public name from kernel.py into this module's namespace.
globals().update({_k: _v for _k, _v in vars(_mod).items() if not _k.startswith("_")})
