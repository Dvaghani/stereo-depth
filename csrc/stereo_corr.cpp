// C++ binding glue for the stereo correlation CUDA kernel.

#include <torch/extension.h>

// Declared in stereo_corr_kernel.cu
void stereo_corr_forward_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    torch::Tensor& out,
    int max_disp);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

void forward(
    torch::Tensor left,
    torch::Tensor right,
    torch::Tensor out,
    int64_t max_disp) {
    CHECK_INPUT(left);
    CHECK_INPUT(right);
    CHECK_INPUT(out);
    TORCH_CHECK(left.sizes() == right.sizes(), "left/right shape mismatch");
    TORCH_CHECK(out.dim() == 4, "out must be 4D");
    TORCH_CHECK(out.size(0) == left.size(0), "batch mismatch");
    TORCH_CHECK(out.size(1) == max_disp, "out disparity dim mismatch");
    TORCH_CHECK(out.size(2) == left.size(2), "height mismatch");
    TORCH_CHECK(out.size(3) == left.size(3), "width mismatch");
    stereo_corr_forward_cuda(left, right, out, static_cast<int>(max_disp));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
          "Stereo correlation volume (forward)",
          py::arg("left"), py::arg("right"), py::arg("out"), py::arg("max_disp"));
}
