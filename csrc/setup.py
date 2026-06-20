"""
Build script for the stereo correlation CUDA extension.

Usage (workstation):
    cd csrc
    python setup.py build_ext --inplace

Usage (Jetson Nano, Maxwell sm_53):
    cd csrc
    TORCH_CUDA_ARCH_LIST="5.3" python setup.py build_ext --inplace

"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="stereo_corr_cuda",
    ext_modules=[
        CUDAExtension(
            name="stereo_corr_cuda",
            sources=["stereo_corr.cpp", "stereo_corr_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                # Fedora 44 ships gcc 16 which nvcc 13.x cannot parse
                # (incompatible libstdc++ 16 / C++23 headers). We install
                # gcc-15 alongside and tell nvcc to use it as the host
                # compiler via -ccbin. For the cxx step you must also
                # export CXX=/usr/bin/g++-15 before building.
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-ccbin=/usr/bin/g++-15",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
