#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "canary_common.cuh"
#include <cstdint>

namespace sglang {

namespace canary {

constexpr uint32_t kCanaryInfoBlockSize = 128;

struct CanaryInfoKernelParams {
  const uint8_t* canary_buf;
  int64_t slot_stride_bytes;
  const int64_t* node_index;
  const int64_t* node_slots;
  const int64_t* node_tokens;
  const float* temperatures;
  const float* top_ps;
  const int32_t* top_ks;
  const float* min_ps;
  const int64_t* sampling_seed;
  int64_t has_seed;
  int64_t* canary_info_out;
  int64_t num_rows;
  int64_t width;
};

SGL_DEVICE uint64_t canary_info_sampling_digest(const CanaryInfoKernelParams& p, int64_t row) {
  uint64_t h = splitmix64(kCanaryChainAnchor ^ kCanaryInfoSamplingDigestSalt);
  h = splitmix64(h ^ static_cast<uint64_t>(__float_as_uint(p.temperatures[row])));
  h = splitmix64(h ^ static_cast<uint64_t>(__float_as_uint(p.top_ps[row])));
  h = splitmix64(h ^ static_cast<uint64_t>(static_cast<int64_t>(p.top_ks[row])));
  h = splitmix64(h ^ static_cast<uint64_t>(__float_as_uint(p.min_ps[row])));
  h = splitmix64(h ^ (p.has_seed != 0 ? static_cast<uint64_t>(p.sampling_seed[row]) : 0ULL));
  h = splitmix64(h ^ (p.has_seed != 0 ? 1ULL : 0ULL));
  return h;
}

__global__ void canary_info_kernel(const CanaryInfoKernelParams __grid_constant__ p) {
  const int64_t flat = static_cast<int64_t>(blockIdx.x) * kCanaryInfoBlockSize + threadIdx.x;
  if (flat >= p.num_rows * p.width) {
    return;
  }
  const int64_t node = p.node_index[flat];
  if (node < 0) {
    p.canary_info_out[flat] = 0;
    return;
  }
  const int64_t slot = p.node_slots[node];
  if (slot < 0) {
    p.canary_info_out[flat] = 0;
    return;
  }
  const int64_t row = flat / p.width;
  const uint64_t prefix = compute_slot_hash(p.canary_buf, p.slot_stride_bytes, slot);
  const uint64_t sampling = canary_info_sampling_digest(p, row);
  const uint64_t token = static_cast<uint64_t>(p.node_tokens[node]);
  p.canary_info_out[flat] = static_cast<int64_t>(splitmix64_mix3(prefix, token, sampling));
}

inline void canary_info_cuda(
    tvm::ffi::TensorView canary_buf,
    tvm::ffi::TensorView node_index,
    tvm::ffi::TensorView node_slots,
    tvm::ffi::TensorView node_tokens,
    tvm::ffi::TensorView temperatures,
    tvm::ffi::TensorView top_ps,
    tvm::ffi::TensorView top_ks,
    tvm::ffi::TensorView min_ps,
    tvm::ffi::TensorView sampling_seed,
    int64_t has_seed,
    tvm::ffi::TensorView canary_info_out) {
  using namespace host;

  SymbolicSize N_slots = {"num_canary_slots"};
  SymbolicSize N_stride = {"slot_stride_bytes"};
  SymbolicSize N_rows = {"num_rows"};
  SymbolicSize N_width = {"width"};
  SymbolicSize N_nodes = {"num_nodes"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();

  TensorMatcher({N_slots, N_stride}).with_dtype<uint8_t>().with_device<kDLGPU>(device_).verify(canary_buf);
  TensorMatcher({N_rows, N_width})
      .with_dtype<int64_t>()
      .with_device<kDLGPU>(device_)
      .verify(node_index)
      .verify(canary_info_out);
  TensorMatcher({N_nodes}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(node_slots).verify(node_tokens);
  TensorMatcher({N_rows})
      .with_dtype<float>()
      .with_device<kDLGPU>(device_)
      .verify(temperatures)
      .verify(top_ps)
      .verify(min_ps);
  TensorMatcher({N_rows}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(top_ks);
  if (has_seed != 0) {
    TensorMatcher({N_rows}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(sampling_seed);
  }

  const int64_t slot_stride_bytes = N_stride.unwrap();
  const int64_t num_rows = N_rows.unwrap();
  const int64_t width = N_width.unwrap();
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      slot_stride_bytes >= static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      "canary_info: slot_stride_bytes must hold at least ",
      static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      " bytes per slot, got ",
      slot_stride_bytes);

  const int64_t total = num_rows * width;
  if (total == 0) {
    return;
  }

  CanaryInfoKernelParams p{};
  p.canary_buf = static_cast<const uint8_t*>(canary_buf.data_ptr());
  p.slot_stride_bytes = slot_stride_bytes;
  p.node_index = static_cast<const int64_t*>(node_index.data_ptr());
  p.node_slots = static_cast<const int64_t*>(node_slots.data_ptr());
  p.node_tokens = static_cast<const int64_t*>(node_tokens.data_ptr());
  p.temperatures = static_cast<const float*>(temperatures.data_ptr());
  p.top_ps = static_cast<const float*>(top_ps.data_ptr());
  p.top_ks = static_cast<const int32_t*>(top_ks.data_ptr());
  p.min_ps = static_cast<const float*>(min_ps.data_ptr());
  p.sampling_seed = has_seed != 0 ? static_cast<const int64_t*>(sampling_seed.data_ptr()) : nullptr;
  p.has_seed = has_seed;
  p.canary_info_out = static_cast<int64_t*>(canary_info_out.data_ptr());
  p.num_rows = num_rows;
  p.width = width;

  const uint32_t grid = static_cast<uint32_t>((total + kCanaryInfoBlockSize - 1) / kCanaryInfoBlockSize);
  LaunchKernel(grid, kCanaryInfoBlockSize, device)(canary_info_kernel, p);
}

}  // namespace canary

}  // namespace sglang
