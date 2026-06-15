# loader.py — turns a raw .cu file into a callable Python module via nvcc.
# Why a loader at all: every kernel needs the same compile-and-wrap dance.
# Writing it once here (mechanism) keeps each kernel file pure math (policy).

import os
from torch.utils.cpp_extension import load_inline

# kernels/ sits next to harness/, one level up from this file
KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels")

_cache = {}   # compiling is slow (~1-2 min first time); never compile the same kernel twice in a session


def load_kernel(name: str, cu_file: str, func_decls: list[str]):
    """
    name       : unique module name — also the build-cache key
    cu_file    : filename inside kernels/, e.g. "01_vector_add.cu"
    func_decls : C++ signatures we expose to Python, e.g.
                 ["torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);"]
    """
    if name in _cache:
        return _cache[name]

    with open(os.path.join(KERNEL_DIR, cu_file)) as f:
        cuda_src = f.read()

    # pull function names out of the declarations so PyTorch knows what to bind
    func_names = [d.split("(")[0].split()[-1] for d in func_decls]

    mod = load_inline(
        name=name,
        cpp_sources="\n".join(func_decls),   # the C++ "header" — what Python may call
        cuda_sources=cuda_src,               # the actual .cu body nvcc compiles
        functions=func_names,
        verbose=False,
        extra_cuda_cflags=["-O3"],           # let nvcc optimize; fair benchmarking needs this
    )
    _cache[name] = mod
    return mod