// Python bindings for the C++ DCT/DST backend (orthel.transforms._r2r).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "cpu/r2r.hpp"

namespace nb = nanobind;

namespace {

using Array = nb::ndarray<nb::c_contig, nb::device::cpu>;

void r2r(const Array& in, Array& out, int axis, const std::string& op,
         bool transpose, int nthreads) {
  if (in.ndim() != out.ndim())
    throw std::invalid_argument("input and output must have the same shape");
  for (std::size_t d = 0; d < in.ndim(); ++d)
    if (in.shape(d) != out.shape(d))
      throw std::invalid_argument("input and output must have the same shape");
  if (in.dtype() != out.dtype())
    throw std::invalid_argument("input and output must have the same dtype");
  if (axis < 0 || static_cast<std::size_t>(axis) >= in.ndim())
    throw std::invalid_argument("axis out of range");

  std::size_t outer = 1, inner = 1;
  for (int d = 0; d < axis; ++d) outer *= in.shape(d);
  for (std::size_t d = axis + 1; d < in.ndim(); ++d) inner *= in.shape(d);
  const std::size_t n = in.shape(axis);
  const orthel::Recipe recipe = orthel::recipe(op, n, transpose);
  const std::size_t threads = nthreads > 0 ? nthreads : 1;

  if (in.dtype() == nb::dtype<double>()) {
    nb::gil_scoped_release release;
    orthel::transform(static_cast<const double*>(in.data()),
                      static_cast<double*>(out.data()), outer, n, inner, recipe,
                      threads);
  } else if (in.dtype() == nb::dtype<float>()) {
    nb::gil_scoped_release release;
    orthel::transform(static_cast<const float*>(in.data()),
                      static_cast<float*>(out.data()), outer, n, inner, recipe,
                      threads);
  } else {
    throw nb::type_error("expected float32 or float64 data");
  }
}

std::size_t max_direct_size(const std::string& op, std::size_t n) {
  return orthel::get_plan<double>(orthel::kernel_of(op), n)->max_direct;
}

}  // namespace

NB_MODULE(_r2r, m) {
  m.doc() = "Fast DCT/DST kernels without an FFT.";
  m.def("r2r", &r2r, nb::arg("input"), nb::arg("output"), nb::arg("axis"),
        nb::arg("op"), nb::arg("transpose"), nb::arg("nthreads"),
        "Apply transform `op` (or its transpose) along `axis` into `output`.");
  m.def("max_direct_size", &max_direct_size, nb::arg("op"), nb::arg("n"),
        "Largest odd-size direct sum in the plan for `op` on `n` points.");
  m.attr("lanes") = orthel::kLanes;
}
