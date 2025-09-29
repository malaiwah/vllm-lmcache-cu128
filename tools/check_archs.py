#!python
import importlib, sys, subprocess, shlex
p = importlib.import_module("lmcache.c_ops").__file__
out = subprocess.check_output(shlex.split(f"cuobjdump --list-elf {p}"), text=True)
assert "sm_89" in out and "sm_120" in out, out
print("OK:", p)
