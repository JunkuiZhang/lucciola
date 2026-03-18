#pragma once

namespace lucciola::kernels {

#define WARP_SIZE 32

template <typename T> union Pack128 {
    float4 f4;
    T elements[16 / sizeof(T)];
};

template <typename T, typename PtrT>
__device__ __forceinline__ Pack128<T> load_128bit(const PtrT *ptr) {
    Pack128<T> pack;
    pack.f4 = *reinterpret_cast<const float4 *>(ptr);
    return pack;
}

template <typename T, typename PtrT>
__device__ __forceinline__ void
store_128bit(PtrT *ptr, const Pack128<T> &pack) {
    *reinterpret_cast<float4 *>(ptr) = pack.f4;
}

} // namespace lucciola::kernels