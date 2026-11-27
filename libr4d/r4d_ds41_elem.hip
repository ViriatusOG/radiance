// R4D -- DeepSeek-V4.1's elem ops. Each kernel rounds where its libref function does.
//
//   token_map            mapped[m] = table[ids[m]]                         (ref_sparse_kv.cpp)
//   stream_gate_add      Engram's normalised-dot gate into hc streams      (ref_sparse_kv.cpp)
//   rope_rows            rotary from a rope_table plane over a head window (ref_attn.cpp)
//   quant_roundtrip      block-scaled fp8 / fp4 quantise-dequantise        (ref_sparse_kv.cpp)
//   mhc_mix              mHC projection, gates and Sinkhorn, pre out       (ref_mhc.cpp)
//   mhc_collapse         x = sum_j pre[j] h[j]                             (ref_mhc.cpp)
//   softmax_pool         channel-wise softmax pooling of ratio rows        (ref_sparse_kv.cpp)
//   entry_store          one entry a block into a paged cache              (ref_sparse_kv.cpp)
//   router_scored_topk   correction-bias routing over named scores         (ref_glm.cpp)
//   embed_lookup_planes  gather from affine code/scale(/table) planes      (ref_gemm.cpp)
//
// Every product and sum is the one libref forms, unfused (-ffp-contract=off), in f32; only the
// order of a long sum differs (a lane's strided partial, then a fixed butterfly). Stores narrow as
// rad_f32_to_bf16 does (RNE, a NaN quieted). libref's transcendentals are glibc's float expf /
// log1pf or ref_exp (exp in double, rounded once); here each is the double routine rounded once,
// which is the correctly rounded value glibc returns in all but near-tie cases.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"
#include "rad_kv_fp8.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

/* ==================================================================== libref's scalar rules */

// rad_f32_to_bf16 (abi/rad_plugin.h): round to nearest even, a NaN quieted rather than rounded.
__device__ __forceinline__ uint16_t ds_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ds_bf(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// rad_f16_to_f32, bit for bit (subnormals renormalised, a NaN's payload kept).
__device__ __forceinline__ float ds_f16(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

// rad_fp8e4m3_to_f32.
__device__ __forceinline__ float ds_e4m3(uint32_t b) {
    const uint32_t s = (b >> 7) << 31;
    const uint32_t e = (b >> 3) & 0xfu;
    uint32_t m = b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

// rad_f32_to_fp8e4m3: RNE, saturating at 448, NaN to 0x7f.
__device__ __forceinline__ uint32_t ds_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;
    if (a >= 0x43e00000u) return s | 0x7eu;
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        uint32_t q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | ((uint32_t)ee << 3) | q;
}

// ref e8m0_to_f32: 2^(e-127), 0xff NaN (e 0 is the subnormal 2^-127).
__device__ __forceinline__ float ds_e8m0(uint32_t e) {
    if (e == 0xffu) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(e ? e << 23 : 0x00400000u);
}

// std::max(a, b) and std::clamp as libstdc++ evaluates them (a NaN `b` is dropped by max, kept by
// clamp), which fmaxf/fminf would not reproduce.
__device__ __forceinline__ float ds_max(float a, float b) { return (a < b) ? b : a; }
__device__ __forceinline__ float ds_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

// The double routines rounded once, as libref's contract states them.
__device__ __forceinline__ float ds_exp_ref(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float ds_log1p_ref(float x) { return (float)log1p((double)x); }

/* The same two values without FP64 on the common path (gfx1201's FP64 rate makes a wave's double
 * exp + log1p cost microseconds, so the router's sqrt(softplus) over 12 experts a lane spent most of
 * its time in FP64). Each is formed
 * in float-float (|error| ~2^-46 for exp, ~2^-40 for log1p, relative), and kept only when every
 * real within a bound of that (2^-43, 2^-38: the double routine's own error is far below) rounds
 * to the same float; otherwise (about one input in 2^17 / 2^13, and outside the reduced ranges)
 * the lane runs the double routine itself. So the float returned is the double routine's, bit for
 * bit, for every input: checked exhaustively on gfx1201 over all 2^32 floats against
 * ds_exp_ref / ds_log1p_ref. Tables: 2^(j/64) as hi + lo;
 * R_j = RN(1/(1 + j/256)) and -log(R_j) as hi + lo. */
__device__ __constant__ uint32_t kDsExpHi[64] = {
    0x3f800000u, 0x3f8164d2u, 0x3f82cd87u, 0x3f843a29u, 0x3f85aac3u, 0x3f871f62u, 0x3f88980fu, 0x3f8a14d5u,
    0x3f8b95c2u, 0x3f8d1adfu, 0x3f8ea43au, 0x3f9031dcu, 0x3f91c3d3u, 0x3f935a2bu, 0x3f94f4f0u, 0x3f96942du,
    0x3f9837f0u, 0x3f99e046u, 0x3f9b8d3au, 0x3f9d3edau, 0x3f9ef532u, 0x3fa0b051u, 0x3fa27043u, 0x3fa43516u,
    0x3fa5fed7u, 0x3fa7cd94u, 0x3fa9a15bu, 0x3fab7a3au, 0x3fad583fu, 0x3faf3b79u, 0x3fb123f6u, 0x3fb311c4u,
    0x3fb504f3u, 0x3fb6fd92u, 0x3fb8fbafu, 0x3fbaff5bu, 0x3fbd08a4u, 0x3fbf179au, 0x3fc12c4du, 0x3fc346cdu,
    0x3fc5672au, 0x3fc78d75u, 0x3fc9b9beu, 0x3fcbec15u, 0x3fce248cu, 0x3fd06334u, 0x3fd2a81eu, 0x3fd4f35bu,
    0x3fd744fdu, 0x3fd99d16u, 0x3fdbfbb8u, 0x3fde60f5u, 0x3fe0ccdfu, 0x3fe33f89u, 0x3fe5b907u, 0x3fe8396au,
    0x3feac0c7u, 0x3fed4f30u, 0x3fefe4bau, 0x3ff28177u, 0x3ff5257du, 0x3ff7d0dfu, 0x3ffa83b3u, 0x3ffd3e0cu,
};
__device__ __constant__ uint32_t kDsExpLo[64] = {
    0x00000000u, 0xb1c43fd0u, 0xb34ea7a9u, 0xb2f14c87u, 0x334f9891u, 0xb352c2e6u, 0xb37eda4bu, 0x336a92deu,
    0xb260aba1u, 0x3336fcb7u, 0xb3697465u, 0x330628cdu, 0x33675624u, 0x32bc4f9cu, 0xb32e0212u, 0x32dc8061u,
    0x33231b71u, 0xb359be90u, 0xb30c5563u, 0xb331a601u, 0x33412342u, 0x31fb9715u, 0x30c3125au, 0xb323ec33u,
    0xb32c9d5eu, 0xb3162d36u, 0xb3162b08u, 0xb314ad82u, 0xb22deaf6u, 0xb3252debu, 0xb37c5aa8u, 0x32154889u,
    0x32cfe77au, 0xb266b974u, 0x330ec5f7u, 0xb31bd983u, 0xb3414fe8u, 0xb3130b1au, 0xb2d6663eu, 0xb2976da2u,
    0x320aa837u, 0xb2dd5119u, 0xb37323a2u, 0xb006c6c2u, 0x3228fc24u, 0xb2944353u, 0xb35c1daau, 0xb3286024u,
    0xb2d4a58au, 0xb2f61d41u, 0xb3504a1cu, 0xb37b43e3u, 0xb21eab59u, 0x33657d15u, 0xb2441be6u, 0x33207898u,
    0xb24116deu, 0x3276cca1u, 0xb348464au, 0x32f167ffu, 0x32292436u, 0x336615a2u, 0xb2923758u, 0x31cf486cu,
};
__device__ __constant__ uint32_t kDsLogR[256] = {
    0x3f800000u, 0x3f7f00ffu, 0x3f7e03f8u, 0x3f7d08e5u, 0x3f7c0fc1u, 0x3f7b1885u, 0x3f7a232du, 0x3f792fb2u,
    0x3f783e10u, 0x3f774e40u, 0x3f76603eu, 0x3f757404u, 0x3f74898du, 0x3f73a0d5u, 0x3f72b9d6u, 0x3f71d48cu,
    0x3f70f0f1u, 0x3f700f01u, 0x3f6f2eb7u, 0x3f6e500fu, 0x3f6d7304u, 0x3f6c9791u, 0x3f6bbdb3u, 0x3f6ae564u,
    0x3f6a0ea1u, 0x3f693965u, 0x3f6865acu, 0x3f679373u, 0x3f66c2b4u, 0x3f65f36du, 0x3f652598u, 0x3f645933u,
    0x3f638e39u, 0x3f62c4a7u, 0x3f61fc78u, 0x3f6135aau, 0x3f607038u, 0x3f5fac1fu, 0x3f5ee95cu, 0x3f5e27ebu,
    0x3f5d67c9u, 0x3f5ca8f1u, 0x3f5beb62u, 0x3f5b2f17u, 0x3f5a740eu, 0x3f59ba42u, 0x3f5901b2u, 0x3f584a5au,
    0x3f579436u, 0x3f56df44u, 0x3f562b81u, 0x3f5578e9u, 0x3f54c77bu, 0x3f541733u, 0x3f53680du, 0x3f52ba08u,
    0x3f520d21u, 0x3f516154u, 0x3f50b6a0u, 0x3f500d01u, 0x3f4f6475u, 0x3f4ebcf9u, 0x3f4e168au, 0x3f4d7127u,
    0x3f4ccccdu, 0x3f4c2978u, 0x3f4b8728u, 0x3f4ae5d8u, 0x3f4a4588u, 0x3f49a634u, 0x3f4907dau, 0x3f486a79u,
    0x3f47ce0cu, 0x3f473294u, 0x3f46980cu, 0x3f45fe74u, 0x3f4565c8u, 0x3f44ce08u, 0x3f443730u, 0x3f43a13eu,
    0x3f430c31u, 0x3f427806u, 0x3f41e4bcu, 0x3f415250u, 0x3f40c0c1u, 0x3f40300cu, 0x3f3fa030u, 0x3f3f112bu,
    0x3f3e82fau, 0x3f3df59du, 0x3f3d6910u, 0x3f3cdd53u, 0x3f3c5264u, 0x3f3bc841u, 0x3f3b3ee7u, 0x3f3ab656u,
    0x3f3a2e8cu, 0x3f39a786u, 0x3f392144u, 0x3f389bc3u, 0x3f381703u, 0x3f379301u, 0x3f370fbbu, 0x3f368d31u,
    0x3f360b61u, 0x3f358a48u, 0x3f3509e7u, 0x3f348a3au, 0x3f340b41u, 0x3f338cfau, 0x3f330f63u, 0x3f32927cu,
    0x3f321643u, 0x3f319ab6u, 0x3f311fd4u, 0x3f30a59bu, 0x3f302c0bu, 0x3f2fb322u, 0x3f2f3adeu, 0x3f2ec33eu,
    0x3f2e4c41u, 0x3f2dd5e6u, 0x3f2d602bu, 0x3f2ceb10u, 0x3f2c7692u, 0x3f2c02b0u, 0x3f2b8f6au, 0x3f2b1cbeu,
    0x3f2aaaabu, 0x3f2a392fu, 0x3f29c84au, 0x3f2957fbu, 0x3f28e83fu, 0x3f287917u, 0x3f280a81u, 0x3f279c7bu,
    0x3f272f05u, 0x3f26c21eu, 0x3f2655c4u, 0x3f25e9f7u, 0x3f257eb5u, 0x3f2513fdu, 0x3f24a9cfu, 0x3f244029u,
    0x3f23d70au, 0x3f236e72u, 0x3f23065eu, 0x3f229ecfu, 0x3f2237c3u, 0x3f21d13au, 0x3f216b31u, 0x3f2105a9u,
    0x3f20a0a1u, 0x3f203c17u, 0x3f1fd80au, 0x3f1f747au, 0x3f1f1166u, 0x3f1eaecdu, 0x3f1e4cadu, 0x3f1deb07u,
    0x3f1d89d9u, 0x3f1d2922u, 0x3f1cc8e1u, 0x3f1c6917u, 0x3f1c09c1u, 0x3f1baadfu, 0x3f1b4c70u, 0x3f1aee73u,
    0x3f1a90e8u, 0x3f1a33cdu, 0x3f19d723u, 0x3f197ae7u, 0x3f191f1au, 0x3f18c3bbu, 0x3f1868c8u, 0x3f180e41u,
    0x3f17b426u, 0x3f175a75u, 0x3f17012eu, 0x3f16a850u, 0x3f164fdau, 0x3f15f7ccu, 0x3f15a025u, 0x3f1548e5u,
    0x3f14f209u, 0x3f149b93u, 0x3f144581u, 0x3f13efd2u, 0x3f139a86u, 0x3f13459cu, 0x3f12f114u, 0x3f129cecu,
    0x3f124925u, 0x3f11f5bdu, 0x3f11a2b4u, 0x3f115009u, 0x3f10fdbcu, 0x3f10abccu, 0x3f105a38u, 0x3f100901u,
    0x3f0fb824u, 0x3f0f67a2u, 0x3f0f177au, 0x3f0ec7abu, 0x3f0e7835u, 0x3f0e2918u, 0x3f0dda52u, 0x3f0d8be3u,
    0x3f0d3dcbu, 0x3f0cf009u, 0x3f0ca29cu, 0x3f0c5584u, 0x3f0c08c1u, 0x3f0bbc51u, 0x3f0b7034u, 0x3f0b246bu,
    0x3f0ad8f3u, 0x3f0a8dcdu, 0x3f0a42f8u, 0x3f09f874u, 0x3f09ae41u, 0x3f09645cu, 0x3f091ac7u, 0x3f08d181u,
    0x3f088889u, 0x3f083fdeu, 0x3f07f781u, 0x3f07af70u, 0x3f0767abu, 0x3f072033u, 0x3f06d905u, 0x3f069223u,
    0x3f064b8au, 0x3f06053cu, 0x3f05bf37u, 0x3f05797cu, 0x3f053408u, 0x3f04eeddu, 0x3f04a9fau, 0x3f04655eu,
    0x3f042108u, 0x3f03dcf9u, 0x3f039930u, 0x3f0355adu, 0x3f03126fu, 0x3f02cf75u, 0x3f028cc0u, 0x3f024a4eu,
    0x3f020821u, 0x3f01c636u, 0x3f01848eu, 0x3f014328u, 0x3f010204u, 0x3f00c122u, 0x3f008081u, 0x3f004020u,
};
__device__ __constant__ uint32_t kDsLogHi[256] = {
    0x00000000u, 0x3b7f8056u, 0x3bff015bu, 0x3c3ee24fu, 0x3c7e0545u, 0x3c9e752fu, 0x3cbdc8d6u, 0x3cdcfe05u,
    0x3cfc14c8u, 0x3d0d86c8u, 0x3d1cf437u, 0x3d2c52dbu, 0x3d3ba2ceu, 0x3d4ae41bu, 0x3d5a16f0u, 0x3d693b59u,
    0x3d785185u, 0x3d83acc1u, 0x3d8b29b9u, 0x3d929fb1u, 0x3d9a0ebau, 0x3da176e7u, 0x3da8d837u, 0x3db032c5u,
    0x3db78694u, 0x3dbed3b5u, 0x3dc61a33u, 0x3dcd5a11u, 0x3dd4936cu, 0x3ddbc63fu, 0x3de2f2a6u, 0x3dea189du,
    0x3df1383au, 0x3df85182u, 0x3dff648au, 0x3e0338a8u, 0x3e06bbf4u, 0x3e0a3c2eu, 0x3e0db958u, 0x3e11337au,
    0x3e14aa96u, 0x3e181ebau, 0x3e1b8fe1u, 0x3e1efe16u, 0x3e22695au, 0x3e25d1b9u, 0x3e29372fu, 0x3e2c99c5u,
    0x3e2ff983u, 0x3e33566au, 0x3e36b07eu, 0x3e3a07cau, 0x3e3d5c48u, 0x3e40ae03u, 0x3e43fd04u, 0x3e474948u,
    0x3e4a92d4u, 0x3e4dd9b3u, 0x3e511de0u, 0x3e545f67u, 0x3e579e49u, 0x3e5ada8cu, 0x3e5e1436u, 0x3e614b44u,
    0x3e647fbdu, 0x3e67b1adu, 0x3e6ae10au, 0x3e6e0de7u, 0x3e71383bu, 0x3e746013u, 0x3e778570u, 0x3e7aa850u,
    0x3e7dc8c6u, 0x3e807362u, 0x3e82012eu, 0x3e838dc3u, 0x3e851928u, 0x3e86a35au, 0x3e882c5fu, 0x3e89b438u,
    0x3e8b3ae5u, 0x3e8cc06au, 0x3e8e44c6u, 0x3e8fc7fdu, 0x3e914a0fu, 0x3e92cb01u, 0x3e944ad0u, 0x3e95c981u,
    0x3e974716u, 0x3e98c38du, 0x3e9a3eeeu, 0x3e9bb934u, 0x3e9d3263u, 0x3e9eaa7cu, 0x3ea02185u, 0x3ea19779u,
    0x3ea30c5du, 0x3ea48034u, 0x3ea5f2fdu, 0x3ea764bbu, 0x3ea8d56cu, 0x3eaa4515u, 0x3eabb3bau, 0x3ead2156u,
    0x3eae8dedu, 0x3eaff984u, 0x3eb16416u, 0x3eb2cdabu, 0x3eb43640u, 0x3eb59dd8u, 0x3eb70476u, 0x3eb86a18u,
    0x3eb9cebfu, 0x3ebb3270u, 0x3ebc952au, 0x3ebdf6f1u, 0x3ebf57c2u, 0x3ec0b7a0u, 0x3ec2168eu, 0x3ec3748du,
    0x3ec4d19du, 0x3ec62dbfu, 0x3ec788f5u, 0x3ec8e33eu, 0x3eca3c9fu, 0x3ecb951au, 0x3eccecacu, 0x3ece4358u,
    0x3ecf991eu, 0x3ed0ee04u, 0x3ed24205u, 0x3ed39524u, 0x3ed4e766u, 0x3ed638c6u, 0x3ed78949u, 0x3ed8d8f1u,
    0x3eda27bdu, 0x3edb75adu, 0x3edcc2c5u, 0x3ede0f04u, 0x3edf5a6du, 0x3ee0a501u, 0x3ee1eebeu, 0x3ee337a7u,
    0x3ee47fbfu, 0x3ee5c702u, 0x3ee70d79u, 0x3ee8531eu, 0x3ee997f4u, 0x3eeadbfbu, 0x3eec1f3au, 0x3eed61abu,
    0x3eeea34fu, 0x3eefe42bu, 0x3ef12441u, 0x3ef2638eu, 0x3ef3a213u, 0x3ef4dfd2u, 0x3ef61ccfu, 0x3ef75906u,
    0x3ef8947au, 0x3ef9cf2du, 0x3efb0921u, 0x3efc4250u, 0x3efd7ac3u, 0x3efeb278u, 0x3effe96fu, 0x3f008fd6u,
    0x3f012a95u, 0x3f01c4f8u, 0x3f025efdu, 0x3f02f8a7u, 0x3f0391f3u, 0x3f042ae4u, 0x3f04c37bu, 0x3f055bb7u,
    0x3f05f397u, 0x3f068b1fu, 0x3f07224cu, 0x3f07b921u, 0x3f084f9eu, 0x3f08e5c2u, 0x3f097b8du, 0x3f0a1100u,
    0x3f0aa61fu, 0x3f0b3ae5u, 0x3f0bcf55u, 0x3f0c6370u, 0x3f0cf735u, 0x3f0d8aa5u, 0x3f0e1dc0u, 0x3f0eb088u,
    0x3f0f42fau, 0x3f0fd51au, 0x3f1066e6u, 0x3f10f860u, 0x3f118987u, 0x3f121a5bu, 0x3f12aadfu, 0x3f133b0fu,
    0x3f13caf0u, 0x3f145a80u, 0x3f14e9c0u, 0x3f1578b0u, 0x3f160750u, 0x3f1695a0u, 0x3f1723a2u, 0x3f17b155u,
    0x3f183ebau, 0x3f18cbd0u, 0x3f19589au, 0x3f19e516u, 0x3f1a7144u, 0x3f1afd27u, 0x3f1b88beu, 0x3f1c1407u,
    0x3f1c9f07u, 0x3f1d29bau, 0x3f1db423u, 0x3f1e3e41u, 0x3f1ec812u, 0x3f1f519du, 0x3f1fdaddu, 0x3f2063d2u,
    0x3f20ec7eu, 0x3f2174e3u, 0x3f21fcfeu, 0x3f2284d2u, 0x3f230c5fu, 0x3f2393a1u, 0x3f241a9fu, 0x3f24a154u,
    0x3f2527c4u, 0x3f25adecu, 0x3f2633ceu, 0x3f26b969u, 0x3f273ec1u, 0x3f27c3d2u, 0x3f28489eu, 0x3f28cd25u,
    0x3f29516au, 0x3f29d569u, 0x3f2a5924u, 0x3f2adc9au, 0x3f2b5fceu, 0x3f2be2c0u, 0x3f2c656du, 0x3f2ce7dau,
    0x3f2d6a01u, 0x3f2debe9u, 0x3f2e6d8eu, 0x3f2eeef3u, 0x3f2f7015u, 0x3f2ff0f6u, 0x3f307197u, 0x3f30f1f8u,
};
__device__ __constant__ uint32_t kDsLogLo[256] = {
    0x00000000u, 0x2dac42f0u, 0x2f310679u, 0x2f70215cu, 0xaff03fc2u, 0xae395a38u, 0x307d5b12u, 0x307af91au,
    0x30678338u, 0x30923d99u, 0x2f7f5ec7u, 0xb08371b2u, 0xaf66916eu, 0x30cc8e3fu, 0x3091971du, 0x2fb6c2d7u,
    0x2d0b153bu, 0x30931c8eu, 0xb175c852u, 0xb15f5ebeu, 0x30c37a3cu, 0xb0cb3722u, 0xb03e79ebu, 0x3153750cu,
    0xb151a94bu, 0xb1784ed3u, 0xafe7325du, 0x30a40676u, 0x3124ad57u, 0xb154f125u, 0xaf2438b8u, 0x3116cb36u,
    0x3162af2eu, 0xb0ddd3a7u, 0x2fedf55eu, 0xb1c35a69u, 0x31cd08e6u, 0x315ce858u, 0x3017b3d4u, 0xb1d75a53u,
    0x3103fea6u, 0x31c647feu, 0xb19e1709u, 0x309b318bu, 0xb166905bu, 0x31534abeu, 0x31343687u, 0xb1967df7u,
    0x2fa793d5u, 0x2f655aa2u, 0x31e19d22u, 0xaeff2116u, 0x31021b21u, 0xb1479500u, 0x31441923u, 0x30a21b51u,
    0x2ff456b8u, 0xb1b82047u, 0xae6a54c7u, 0xaff7d0e4u, 0xb16fe800u, 0xb1bf04a5u, 0xb069398au, 0x3190c18du,
    0x31735345u, 0xb1df63a1u, 0x30d3eee6u, 0xb0d504ddu, 0xb06a868du, 0xb1f55a5au, 0xb088e9b0u, 0x31ecb7b5u,
    0xb0a8706fu, 0xb1ea55fcu, 0xb214b2fbu, 0x304d590du, 0x3227390fu, 0xb119522cu, 0xb2651b52u, 0xb1bd8aeau,
    0xb205459fu, 0xb23abdc9u, 0xb2496660u, 0x311228bdu, 0x31d9ef2du, 0xb262089cu, 0x323de86au, 0xb248762du,
    0xb08f7168u, 0x31828832u, 0xb256782au, 0x311ff619u, 0xb1d2d743u, 0xb0bf0472u, 0xb25f9582u, 0x32784e81u,
    0x310717b3u, 0x326c009du, 0xb2488876u, 0xb1fbffb1u, 0xb17481f5u, 0x3226dd26u, 0xb213aa59u, 0x314d33f9u,
    0x31ab013au, 0x326a793eu, 0x32139c7au, 0x3160f87du, 0x3166c52du, 0x31b6b288u, 0x31f5743du, 0xb26876ddu,
    0x312ef402u, 0xb151890bu, 0x31bba8f5u, 0x30fbd7e2u, 0xb15f540fu, 0xb19be56cu, 0x31a3d163u, 0x31b66174u,
    0x320c1426u, 0xb1a4e96eu, 0x31f6cc5au, 0xb03a57b3u, 0x32621171u, 0x31a26a2bu, 0xb21e81f7u, 0xb25a3f7au,
    0x324bf985u, 0xb1dd48cdu, 0x323e5bb1u, 0xb1ba84c2u, 0xb18cbef6u, 0x31b3161du, 0xb2738879u, 0xb208203au,
    0xb263370au, 0xb1c86194u, 0x32493110u, 0x31736b43u, 0xb0ac7242u, 0xb1a541f9u, 0xb24b252bu, 0x3223f700u,
    0xafcacbb4u, 0x327cd7adu, 0x3190be7eu, 0xb20cde1au, 0x32700ebdu, 0x32262c69u, 0xb1feba18u, 0xb2593a6bu,
    0xb09f1daau, 0x32146d5cu, 0xb1d0727bu, 0xb2717890u, 0xb2029787u, 0xb20a79e0u, 0x30468cc2u, 0xb27ff220u,
    0xb2050f90u, 0x318023ccu, 0x30f8145cu, 0x325be9edu, 0x31cf1e65u, 0xb25c9ea8u, 0x2fafe94au, 0xb2f7a007u,
    0xb1968bc0u, 0x3225cc68u, 0xb2a16d93u, 0xb2c79ba6u, 0x32d1cdf4u, 0xb1e52ebcu, 0xb2ea8caeu, 0xb2f11eb3u,
    0x2f94aa0cu, 0xb2fe6294u, 0x324a3991u, 0xb1325b41u, 0xb2ad2b36u, 0xb2ff8330u, 0x32bd3efcu, 0x32ec43d5u,
    0x31fd357cu, 0x320974c2u, 0x32131335u, 0x31cd73c2u, 0x326cf92eu, 0x32ffda2au, 0x320e2786u, 0xb2031200u,
    0x324e081bu, 0xb2031ed6u, 0x3212556eu, 0x3102df76u, 0xb2482818u, 0x32fc8393u, 0x32131efau, 0x3285430fu,
    0x32e89b12u, 0x3276c0f0u, 0xb29bd4f7u, 0xb1be6fd4u, 0x323891d3u, 0xb23a246cu, 0xb289bfd9u, 0x327612a9u,
    0xb2ab0ef4u, 0x3210ffceu, 0xb23dbc52u, 0x30b28995u, 0x31c73878u, 0x3119b751u, 0x3248e8f7u, 0x31ae1e63u,
    0xb2da9d5eu, 0x323bd7bfu, 0x31ffff0bu, 0xb27119beu, 0x32ef1570u, 0x323d65ecu, 0xb2d6e90cu, 0xb1d7c8c5u,
    0x32c4672cu, 0x32a1f80cu, 0x325739ceu, 0x32667761u, 0xb26c7427u, 0x32b53aabu, 0x318aa239u, 0xb2efd03fu,
    0xb209f827u, 0xb2a291d9u, 0x32619dc8u, 0xb00c05e9u, 0x3236eb77u, 0x32c3026fu, 0xb13f4bc9u, 0x3287a345u,
    0xb29a43a5u, 0xb26ccc90u, 0xb20ccd6cu, 0x32dd40bdu, 0xb1f30317u, 0xb2be90e7u, 0x3239af32u, 0xb19145deu,
    0x32d359f4u, 0x2f9c71deu, 0x325d82edu, 0xb2feb66eu, 0x32b24167u, 0x32723124u, 0xb2b78771u, 0x31523233u,
};

__device__ __forceinline__ void ds_two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void ds_two_prod(float a, float b, float& p, float& e) {
    p = a * b;
    e = fmaf(a, b, -p);
}
// Every real within rel*zh of zh + zl (zh > 0, normal) rounds to zh.
__device__ __forceinline__ bool ds_settled(float zh, float zl, float rel) {
    const uint32_t u = __float_as_uint(zh);
    const float nb = __uint_as_float(zl < 0.0f ? u - 1u : u + 1u);
    return fabsf(zl) + rel * zh < fabsf(nb - zh) * 0.5f;
}
__device__ __forceinline__ float ds_tab(const uint32_t* t, int i) { return __uint_as_float(t[i]); }

// Branch-free, so a caller's independent evaluations interleave; false = take the double routine.
__device__ __forceinline__ bool ds_exp_fast(float xi, float& y) {
    const bool dom = fabsf(xi) <= 87.0f;
    const float x = dom ? xi : 0.0f;
    const float kf = rintf(x * __uint_as_float(0x42b8aa3bu));     // x * 64 / ln2
    const int k = (int)kf;
    const float c1 = __uint_as_float(0x3c316000u), c2 = __uint_as_float(0x3690bfbfu),
                c3 = __uint_as_float(0xaa630865u);                 // ln2/64 = c1 + c2 + c3
    const float rh0 = x - kf * c1;                                 // both steps exact
    float ph, pl, rh, rl;
    ds_two_prod(kf, c2, ph, pl);
    ds_two_sum(rh0, -ph, rh, rl);
    rl = rl - (pl + kf * c3);
    ds_two_sum(rh, rl, rh, rl);                                    // r = rh + rl, |r| <= ln2/128
    float qh, ql, eh, el;
    ds_two_prod(rh, rh, qh, ql);
    const float t = rh * qh * (0.16666667f + rh * (0.041666668f + rh * 0.008333334f));
    ds_two_sum(rh, 0.5f * qh, eh, el);                             // e^r - 1 = eh + el
    el = el + (t + (0.5f * ql + (rh * rl + rl)));
    const float th = ds_tab(kDsExpHi, k & 63), tl = ds_tab(kDsExpLo, k & 63);
    float p2h, p2l, yh, yl, zh, zl;
    ds_two_prod(th, eh, p2h, p2l);
    ds_two_sum(th, p2h, yh, yl);
    yl = yl + (tl + (p2l + (th * el + tl * eh)));
    ds_two_sum(yh, yl, zh, zl);
    y = ldexpf(zh, k >> 6);
    return dom & ds_settled(zh, zl, 0x1p-43f);  // `&`: no branch
}

__device__ __forceinline__ bool ds_log1p_fast(float xi, float& y) {
    const bool dom = xi >= 0x1p-60f && xi <= 0x1p100f;
    const float x = dom ? xi : 1.0f;
    float uh, ul;
    ds_two_sum(1.0f, x, uh, ul);                                   // 1 + x = uh + ul exactly
    const uint32_t bu = __float_as_uint(uh);
    const int m = (int)(bu >> 23) - 127, j = (int)((bu >> 15) & 255u);
    const float f = __uint_as_float((bu & 0x007fffffu) | 0x3f800000u), r = ds_tab(kDsLogR, j);
    float ph, pl, sh, sl;
    ds_two_prod(f, r, ph, pl);
    ds_two_sum(ph - 1.0f, pl, sh, sl);                             // f r - 1, ph - 1 exact
    sl = sl + ldexpf(ul, -m) * r;                                  // s = (1 + x) 2^-m r - 1
    ds_two_sum(sh, sl, sh, sl);
    float qh, ql, lh, ll;
    ds_two_prod(sh, sh, qh, ql);
    ql = ql + 2.0f * sh * sl;
    const float p3 = sh * qh *
        (0.33333334f + sh * (-0.25f + sh * (0.2f + sh * (-0.16666667f + sh * 0.14285715f))));
    ds_two_sum(sh, -0.5f * qh, lh, ll);                            // log1p(s) = lh + ll
    ll = ll + (p3 + (sl - 0.5f * ql));
    const float mf = (float)m, l2h = __uint_as_float(0x3f317200u),
                l2l = __uint_as_float(0x35bfbe8eu), l2l2 = __uint_as_float(0x29779abdu);
    float bh, bl, a0, e0, a1, e1, zh, zl;
    ds_two_prod(mf, l2l, bh, bl);
    ds_two_sum(mf * l2h, ds_tab(kDsLogHi, j), a0, e0);             // m ln2 + -log(R_j) + log1p(s)
    ds_two_sum(a0, lh, a1, e1);
    const float lo = e1 + (e0 + (ll + (bh + (ds_tab(kDsLogLo, j) + (bl + mf * l2l2)))));
    ds_two_sum(a1, lo, zh, zl);
    y = zh;
    return dom & ds_settled(zh, zl, 0x1p-38f);
}

__device__ __forceinline__ float ds_exp(float x) {
    float y;
    return ds_exp_fast(x, y) ? y : ds_exp_ref(x);
}
// N independent values in place: every fast path first, then the rare double routines.
// (One branch for all N, so the N fast paths interleave ahead of it.)
template <int N>
__device__ __forceinline__ void ds_exp_n(float (&v)[N]) {
    float y[N];
    bool ok[N], all = true;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        ok[i] = ds_exp_fast(v[i], y[i]);
        all &= ok[i];
    }
    if (!all)
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (!ok[i]) y[i] = ds_exp_ref(v[i]);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = y[i];
}
template <int N>
__device__ __forceinline__ void ds_log1p_n(float (&v)[N]) {
    float y[N];
    bool ok[N], all = true;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        ok[i] = ds_log1p_fast(v[i], y[i]);
        all &= ok[i];
    }
    if (!all)
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (!ok[i]) y[i] = ds_log1p_ref(v[i]);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = y[i];
}

// An element of a float plane of any of the dtypes ld_dt reads that these ops accept.
__device__ __forceinline__ float ds_ld(uint32_t dt, const void* p, long i) {
    switch (dt) {
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F16:    return ds_f16(((const uint16_t*)p)[i]);
        case RAD_BF16:   return ds_bf(((const uint16_t*)p)[i]);
        case RAD_F8E4M3: return ds_e4m3(((const uint8_t*)p)[i]);
        case RAD_E8M0:   return ds_e8m0(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        case RAD_U8:     return (float)((const uint8_t*)p)[i];
        default:         return 0.0f;
    }
}

/* ==================================================================== host-side operand checks */

// An index operand read as libref's offlin(t, m): element m at m * step. Contiguous, or one
// non-unit axis with its stride.
bool lin_step(const RadTensor& t, long* step) {
    if (!t.data || t.rank == 0) return false;
    if (rad_tensor_is_contiguous(&t)) { *step = 1; return true; }
    int nd = 0, d = 0;
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.shape[i] != 1) { ++nd; d = (int)i; }
    if (nd != 1) return false;
    *step = (long)t.stride[d];
    return true;
}
// [rows, cols] with a unit inner stride; the row pitch is stride[0].
bool mat(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols &&
           t.stride[1] == 1 && t.stride[0] >= cols;
}
// [rows, cols] of any inner stride (the few f32 coefficients a row of mhc_mix writes).
bool mat_s(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}
long last_stride(const RadTensor& t) { return t.rank ? (long)t.stride[t.rank - 1] : 1; }
unsigned grid_rows(long rows) { return (unsigned)(rows < 65535 ? rows : 65535); }

/* ==================================================================== token_map */

__global__ __launch_bounds__(256) void token_map_kernel(const int32_t* ids, long is,
                                                        const int32_t* tab, long ts, int32_t* y,
                                                        long ys, long M, long nv) {
    for (long m = (long)blockIdx.x * 256 + threadIdx.x; m < M; m += (long)gridDim.x * 256) {
        const long id = ids[m * is];
        y[m * ys] = id < 0 || id >= nv ? -1 : tab[id * ts];
    }
}

int token_map_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &tab = a->t[1], &y = a->t[2];
    const long nv = (long)r4d_geti_or(a, "n_vocab", 0);
    const long M = (long)rad_tensor_numel(&x);
    if (nv < 1 || rad_tensor_numel(&tab) != nv || M < 1 || rad_tensor_numel(&y) != M)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_I32 || tab.dtype != RAD_I32 || y.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    long is, ts, ys;
    if (!lin_step(x, &is) || !lin_step(tab, &ts) || !lin_step(y, &ys)) return R4D_NO(RAD_E_SHAPE);
    const long blocks = (M + 255) / 256;
    return R4D_EMIT(stream, token_map_kernel, dim3((unsigned)(blocks < 65535 ? blocks : 65535)),
                    dim3(256), 0, (const int32_t*)x.data, is, (const int32_t*)tab.data, ts,
                    (int32_t*)y.data, ys, M, nv);
}

/* ==================================================================== stream_gate_add
 * One workgroup a (row, stream): the three sums over the stream's n values (a lane's strided
 * partial, the wave butterfly, the eight waves in order), the gate on one lane, then the write. */

__global__ __launch_bounds__(256) void stream_gate_kernel(const uint16_t* h, long hp,
                                                          const uint16_t* k, long kp,
                                                          const uint16_t* v, long vp,
                                                          const float* w, long wst, uint16_t* out,
                                                          long op, int n, long rows, float eps,
                                                          float ge, const int32_t* mask) {
    __shared__ float red[3][8];
    __shared__ float gate_s;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, s = blockIdx.x;
    const float* ws = w + (long)s * n * wst;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        /* Image-span rows (mask != 0) take no Engram update: the reference zeros that gate. */
        if (mask && mask[r]) {
            const uint16_t* src = h + r * hp + (long)s * n;
            uint16_t* dst = out + r * op + (long)s * n;
            for (int i = tid; i < n; i += 256) dst[i] = src[i];
            continue;
        }
        const uint16_t* hr = h + r * hp + (long)s * n;
        const uint16_t* kr = k + r * kp + (long)s * n;
        float sh = 0, sk = 0, dot = 0;
        for (int i = tid; i < n; i += 256) {
            const float hv = ds_bf(hr[i]), kv = ds_bf(kr[i]);
            sh += hv * hv;
            sk += kv * kv;
            dot += hv * ws[(long)i * wst] * kv;
        }
        sh = r4d_wave_sum(sh);
        sk = r4d_wave_sum(sk);
        dot = r4d_wave_sum(dot);
        if (!lane) {
            red[0][wave] = sh;
            red[1][wave] = sk;
            red[2][wave] = dot;
        }
        __syncthreads();
        if (!tid) {
            float a = 0, b = 0, c = 0;
            for (int q = 0; q < 8; ++q) {
                a += red[0][q];
                b += red[1][q];
                c += red[2][q];
            }
            const float rstd =
                (1.0f / sqrtf(a / (float)n + eps)) * (1.0f / sqrtf(b / (float)n + eps));
            c = c * rstd * (1.0f / sqrtf((float)n));
            const float t = copysignf(sqrtf(ds_max(fabsf(c), ge)), c);
            gate_s = 1.0f / (1.0f + ds_exp(-t));
        }
        __syncthreads();
        const float g = gate_s;
        const uint16_t* vr = v + r * vp;
        uint16_t* orow = out + r * op + (long)s * n;
        for (int i = tid; i < n; i += 256) orow[i] = ds_bf16(ds_bf(hr[i]) + g * ds_bf(vr[i]));
        __syncthreads();
    }
}

int stream_gate_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &k = a->t[1], &v = a->t[2], &w = a->t[3], &out = a->t[4];
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    const float eps = (float)r4d_getf(a, "eps", -1.0), ge = (float)r4d_getf(a, "gate_eps", -1.0);
    if (n < 1 || hc < 1 || hc > 16 || n * hc > INT_MAX || !(eps >= 0) || !(ge >= 0) ||
        h.rank != 2 || h.shape[0] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || k.dtype != RAD_BF16 || v.dtype != RAD_BF16 || w.dtype != RAD_F32 ||
        out.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat(k, RAD_BF16, rows, hc * n) ||
        !mat(v, RAD_BF16, rows, n) || !mat(out, RAD_BF16, rows, hc * n) || !w.data ||
        rad_tensor_numel(&w) != hc * n || !(w.rank == 1 || rad_tensor_is_contiguous(&w)))
        return R4D_NO(RAD_E_SHAPE);
    const int32_t* mask = a->n_t >= 6 && a->t[5].data ? (const int32_t*)a->t[5].data : nullptr;
    return R4D_EMIT(stream, stream_gate_kernel, dim3((unsigned)hc, grid_rows(rows)), dim3(256), 0,
                    (const uint16_t*)h.data, (long)h.stride[0], (const uint16_t*)k.data,
                    (long)k.stride[0], (const uint16_t*)v.data, (long)v.stride[0],
                    (const float*)w.data, last_stride(w), (uint16_t*)out.data, (long)out.stride[0],
                    (int)n, rows, eps, ge, mask);
}

/* ==================================================================== rope_rows
 * One thread a rotated pair of a row, rows over grid.y. */

__global__ __launch_bounds__(256) void rope_rows_kernel(uint16_t* x, long xp, const float* cs,
                                                        long cp, long cst, const int32_t* pos,
                                                        long ps, long M, long tab_rows, int hd,
                                                        int half, int off, int pairs, int gptj,
                                                        float sign) {
    const int q = blockIdx.x * 256 + threadIdx.x;
    if (q >= pairs) return;
    const int hh = q / half, i = q - hh * half;
    const long ia = (long)hh * hd + off + (gptj ? 2 * i : i), ib = gptj ? ia + 1 : ia + half;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long p = pos[m * ps];
        if (p < 0 || p >= tab_rows) continue;
        const float c = cs[p * cp + (long)i * cst], s = sign * cs[p * cp + (long)(half + i) * cst];
        uint16_t* row = x + m * xp;
        const float va = ds_bf(row[ia]), vb = ds_bf(row[ib]);
        row[ia] = ds_bf16(va * c - vb * s);
        row[ib] = ds_bf16(va * s + vb * c);
    }
}

int rope_rows_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    const long nh = (long)r4d_geti_or(a, "n_head", 0), hd = (long)r4d_geti_or(a, "head_dim", 0);
    const long rot = (long)r4d_geti_or(a, "rot", 0), off = (long)r4d_geti_or(a, "off", -1);
    const long inv = (long)r4d_geti_or(a, "inverse", -1);
    const char* mode = r4d_gets(a, "mode");
    if (!mode) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1) || nh * hd > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || cs.dtype != RAD_F32 || pos.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = x.shape[0];
    long ps;
    if (!mat(x, RAD_BF16, M, nh * hd) || cs.rank != 2 || cs.shape[1] != rot ||
        rad_tensor_numel(&pos) != M || !lin_step(pos, &ps))
        return R4D_NO(RAD_E_SHAPE);
    const long half = rot / 2, pairs = nh * half;
    return R4D_EMIT(stream, rope_rows_kernel, dim3((unsigned)((pairs + 255) / 256), grid_rows(M)),
                    dim3(256), 0, (uint16_t*)x.data, (long)x.stride[0], (const float*)cs.data,
                    (long)cs.stride[0], (long)cs.stride[1], (const int32_t*)pos.data, ps, M,
                    (long)cs.shape[0], (int)hd, (int)half, (int)off, (int)pairs, gptj ? 1 : 0,
                    inv ? -1.0f : 1.0f);
}

/* ==================================================================== quant_roundtrip
 * A block's amax, its scale and the round trip of each value, as ref_quant_roundtrip's
 * block_scale and roundtrip. A power-of-two block up to 32 is a lane group (one value a thread, the
 * max over the group's lanes); any other block is one thread a block. amax is a max, so both give
 * the same bits. */

enum { kFp8 = 0, kFp4 = 1 };
enum { kUe8m0 = 0, kE4m3 = 1, kScaleF32 = 2 };

__device__ __forceinline__ int ds_ceil_log2(float v) {
    const uint32_t u = __float_as_uint(v);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
__device__ __forceinline__ float ds_pow2(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }

__device__ __forceinline__ float block_scale(float amax, int fmt, int sc) {
    if (fmt == kFp8) {
        amax = ds_max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
    }
    if (sc == kE4m3) {
        amax = ds_max(amax, 6.0f * 0x1p-9f);
        return ds_e4m3(ds_to_e4m3(amax / 6.0f));
    }
    amax = ds_max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
}

// ref e2m1_code then mxfp4_elem_to_f32: the nearest of the eight magnitudes, ties to the even
// code, the sign kept (so -0 for a small negative), NaN to +0.
__device__ __forceinline__ float ds_e2m1_rt(float v) {
    if (v != v) return 0.0f;
    const bool neg = (__float_as_uint(v) >> 31) != 0;
    const float a = fabsf(v);
    const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float bm = 0.0f;
#pragma unroll
    for (int c = 1; c < 8; ++c) {
        const float d = fabsf(mag[c] - a), db = fabsf(bm - a);
        if (d < db || (d == db && (c & 1) == 0)) bm = mag[c];
    }
    return neg ? -bm : bm;
}

__device__ __forceinline__ float roundtrip(float v, float scale, int fmt) {
    if (fmt == kFp8) return ds_e4m3(ds_to_e4m3(ds_clamp(v / scale, -448.0f, 448.0f))) * scale;
    return ds_e2m1_rt(ds_clamp(v / scale, -6.0f, 6.0f)) * scale;
}

template <int B>
__global__ __launch_bounds__(256) void quant_group_kernel(uint16_t* x, long xp, int n, long rows,
                                                          int fmt, int sc) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* row = x + r * xp;
        const float v = in ? ds_bf(row[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;  // std::max(amax, NaN) keeps amax: a NaN never wins
        const float amax = r4d_xor_max<B>(a);
        if (in) row[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

__global__ __launch_bounds__(256) void quant_serial_kernel(uint16_t* x, long xp, int n, int block,
                                                           long rows, int fmt, int sc) {
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= n / block) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* p = x + r * xp + (long)b * block;
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = ds_max(amax, fabsf(ds_bf(p[i])));
        const float s = block_scale(amax, fmt, sc);
        for (int i = 0; i < block; ++i) p[i] = ds_bf16(roundtrip(ds_bf(p[i]), s, fmt));
    }
}

int quant_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 1 || !a->t[0].data) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const long n = (long)r4d_geti_or(a, "n", 0), block = (long)r4d_geti_or(a, "block", 0);
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "scale");
    if (!fs || !ss) return R4D_NO(RAD_E_UNSUPPORTED);
    int fmt, sc;
    if (!std::strcmp(fs, "fp8_e4m3")) fmt = kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) fmt = kFp4;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (!std::strcmp(ss, "ue8m0")) sc = kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && fmt == kFp4) sc = kE4m3;
    else if (!std::strcmp(ss, "f32") && fmt == kFp8) sc = kScaleF32;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (n < 1 || block < 1 || n % block || n > INT_MAX || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long rows = x.shape[0];
    if (!mat(x, RAD_BF16, rows, n)) return R4D_NO(RAD_E_SHAPE);
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    const dim3 grid((unsigned)((n + 255) / 256), grid_rows(rows));
#define QRT_GROUP(B)                                                                               \
    case B:                                                                                        \
        return R4D_EMIT(stream, quant_group_kernel<B>, grid, dim3(256), 0, xd, xp, (int)n, rows,   \
                        fmt, sc)
    switch (block) {
        QRT_GROUP(1);
        QRT_GROUP(2);
        QRT_GROUP(4);
        QRT_GROUP(8);
        QRT_GROUP(16);
        QRT_GROUP(32);
        default: break;
    }
#undef QRT_GROUP
    const long nb = n / block;
    return R4D_EMIT(stream, quant_serial_kernel, dim3((unsigned)((nb + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, xd, xp, (int)n, (int)block, rows, fmt, sc);
}

/* ==================================================================== mhc_mix
 * The projection is mhc_pre's: hc*(hc+2) dots of the row with fn and its sum of squares. The row
 * splits into 1024-column chunks; in a chunk each sum is lane L's columns L, L+32, .. in order,
 * closed with the wave butterfly, and the chunk partials are added in chunk order. With scratch
 * the chunks are separate workgroups whose waves split the sums (mix_split_kernel), then a
 * finishing wave a row; without, one wave a row walks the chunks itself, for the same bits. The
 * finish is ref finish_row lane-parallel: element (j,k) of the Sinkhorn matrix on lane j*hc+k (and
 * +32 from hc 6), every row and column sum gathered by shuffles in the scalar loop's order. */

constexpr int kMixCols = 1024;

struct MixOut {
    const float* scale;
    long sst;
    const float* base;
    long bst;
    float* pre;
    long prp, prs;
    float* post;
    long pop, pos;
    float* comb;
    long cp, cs;
    int width, iters;
    float eps, heps;
};

template <int HC, int R>
__device__ __forceinline__ void mix_chunk(const uint16_t* h, long hp, const float* fn, long fp,
                                          int width, long r0, int nr, int chunk, int lane,
                                          float (&acc)[HC * (HC + 2) + 1][R]) {
    constexpr int MIX = HC * (HC + 2);
#pragma unroll
    for (int j = 0; j <= MIX; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[j][r] = 0.0f;
    const int c0 = chunk * kMixCols;
    for (int t = 0; t < kMixCols / 32; ++t) {
        const int col = c0 + t * 32 + lane;
        const bool in = col < width;
        float v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = in && r < nr ? ds_bf(h[(r0 + r) * hp + col]) : 0.0f;
#pragma unroll
        for (int j = 0; j < MIX; ++j) {
            const float w = in ? fn[(long)j * fp + col] : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) acc[j][r] += v[r] * w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[MIX][r] += v[r] * v[r];
    }
}

// The scales and biases a lane of mix_finish reads, loaded before the partials so their latencies
// overlap.
template <int HC>
struct MixPre {
    float s0, s1, s2, bpre, bpost, bc[(HC * HC + 31) / 32];
};
template <int HC>
__device__ __forceinline__ MixPre<HC> mix_pre(int lane, const MixOut& o) {
    constexpr int NN = HC * HC, NE = (NN + 31) / 32;
    MixPre<HC> p;
    p.s0 = o.scale[0];
    p.s1 = o.scale[o.sst];
    p.s2 = o.scale[2 * o.sst];
    const int jl = lane < HC ? lane : 0;
    p.bpre = o.base[(long)jl * o.bst];
    p.bpost = o.base[(long)(HC + jl) * o.bst];
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u;
        p.bc[u] = o.base[(long)(2 * HC + (e < NN ? e : 0)) * o.bst];
    }
    return p;
}

// d[q] holds sum j = lane + 32q (the hc*(hc+2) dots, then the sum of squares) in every lane that
// owns one. Every lane must enter: the gathers are shuffles.
template <int HC>
__device__ __forceinline__ void mix_finish(const float (&d)[(HC * (HC + 2) + 32) / 32], int lane,
                                           long row, const MixOut& o, const MixPre<HC>& pr) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32, NN = HC * HC, NE = (NN + 31) / 32;
    const float sq = __shfl(d[MIX / 32], MIX % 32, 32);
    const float inv = 1.0f / sqrtf(sq / (float)o.width + o.eps);
    float cf[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) cf[q] = d[q] * inv;
    auto coeff = [&](int j) {
        float r = 0.0f;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float t = __shfl(cf[q], j & 31, 32);
            if ((j >> 5) == q) r = t;
        }
        return r;
    };
    const float s0 = pr.s0, s1 = pr.s1, s2 = pr.s2;
    const int jl = lane < HC ? lane : 0;
    const float cpre = coeff(jl), cpost = coeff(HC + jl);
    const float zpre = -(cpre * s0 + pr.bpre), z = cpost * s1 + pr.bpost;
    float c[NE];
    int rb[NE], ck[NE];
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u, ee = e < NN ? e : 0;
        rb[u] = ee / HC * HC;
        ck[u] = ee % HC;
        const float cq = coeff(2 * HC + ee);
        c[u] = e < NN ? cq * s2 + pr.bc[u] : 0.0f;
    }
    auto get = [&](int idx) {
        float r = 0.0f;
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            const float t = __shfl(c[u], idx & 31, 32);
            if ((idx >> 5) == u) r = t;
        }
        return r;
    };
    {
        float mx[NE], sum[NE];
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            mx[u] = -INFINITY;
#pragma unroll
            for (int k = 0; k < HC; ++k) mx[u] = ds_max(mx[u], get(rb[u] + k));
        }
        float ex[2 + NE];  // every lane's three exps in one batch (pre and post kept on hc lanes)
        ex[0] = zpre;
        ex[1] = -z;
#pragma unroll
        for (int u = 0; u < NE; ++u) ex[2 + u] = c[u] - mx[u];
        ds_exp_n(ex);
        if (lane < HC) {
            const float pj = 1.0f / (1.0f + ex[0]) + o.heps;
            o.post[row * o.pop + lane * o.pos] = 2.0f / (1.0f + ex[1]);
            o.pre[row * o.prp + lane * o.prs] = pj;
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = ex[2 + u];
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = c[u] / sum[u] + o.heps;
    }
    for (int it = 0; it < o.iters; ++it) {
        float sum[NE];
        if (it) {
#pragma unroll
            for (int u = 0; u < NE; ++u) {
                sum[u] = 0.0f;
#pragma unroll
                for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
            }
#pragma unroll
            for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int j = 0; j < HC; ++j) sum[u] += get(j * HC + ck[u]);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
    }
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u;
        if (e < NN) o.comb[row * o.cp + (long)e * o.cs] = c[u];
    }
}

// The same partials at decode heights: the WV waves of a workgroup take the same R rows of one
// chunk and split its hc*(hc+2) dots between them (wave w forms dots w, w+WV, ..; wave
// hc*(hc+2) % WV also the sum of squares), each walked as mix_chunk walks it (lane L the columns
// L, L+32, .. in order, each product added unfused), so a partial is the same sum in the same
// order whichever wave forms it, for any R, WV and row count. A wave's loads for TS column steps
// go out before their adds: with a chunk a workgroup and the rows over its waves, one wave issues a
// row's 24 loads a step and waits on them, and at decode heights those load-bound steps were nearly
// the whole call.
constexpr int kMixSteps = 8;

template <int HC, int R, int WV, int TS>
__global__ __launch_bounds__(32 * WV) void mix_split_kernel(const uint16_t* h, long hp,
                                                            const float* fn, long fp, float* part,
                                                            int width, int chunks, long rows) {
    constexpr int MIX = HC * (HC + 2), S = MIX + 1, JW = (MIX + WV - 1) / WV;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, chunk = blockIdx.x;
    const int c0 = chunk * kMixCols;
    const float* wr[JW];  // a slot past the last dot walks row 0 and is dropped
#pragma unroll
    for (int q = 0; q < JW; ++q) {
        const int j = wave + WV * q;
        wr[q] = fn + (long)(j < MIX ? j : 0) * fp;
    }
    for (long g = blockIdx.y; g * R < rows; g += gridDim.y) {
        const long r0 = g * R;
        const int nr = (int)(rows - r0 < R ? rows - r0 : R);
        const uint16_t* hr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) hr[r] = h + (r0 + (r < nr ? r : 0)) * hp;
        float acc[JW + 1][R];  // [JW]: the sum of squares
#pragma unroll
        for (int q = 0; q <= JW; ++q)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[q][r] = 0.0f;
        auto walk = [&](auto full, auto sq) {
            constexpr bool FULL = decltype(full)::value, SQ = decltype(sq)::value;
#pragma unroll 1  // unrolled, the loads of every step went up front and spilled
            for (int t0 = 0; t0 < kMixCols / 32; t0 += TS) {
                float v[TS][R], w[TS][JW];
#pragma unroll
                for (int u = 0; u < TS; ++u) {
                    const int col = c0 + (t0 + u) * 32 + lane;
                    const bool in = FULL || col < width;
                    const int cc = in ? col : 0;  // a valid address; the value is replaced by 0
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float x = ds_bf(hr[r][cc]);
                        v[u][r] = in && r < nr ? x : 0.0f;
                    }
#pragma unroll
                    for (int q = 0; q < JW; ++q) {
                        const float x = wr[q][cc];
                        w[u][q] = in ? x : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < TS; ++u) {
#pragma unroll
                    for (int q = 0; q < JW; ++q)
#pragma unroll
                        for (int r = 0; r < R; ++r) acc[q][r] += v[u][r] * w[u][q];
                    if (SQ)
#pragma unroll
                        for (int r = 0; r < R; ++r) acc[JW][r] += v[u][r] * v[u][r];
                }
            }
        };
        const bool full = c0 + kMixCols <= width, sq = wave == MIX % WV;
        if (full) {
            if (sq) walk(std::true_type{}, std::true_type{});
            else walk(std::true_type{}, std::false_type{});
        } else {
            if (sq) walk(std::false_type{}, std::true_type{});
            else walk(std::false_type{}, std::false_type{});
        }
#pragma unroll
        for (int q = 0; q <= JW; ++q) {
            const int j = q < JW ? wave + WV * q : MIX;
            if (q < JW ? j >= MIX : !sq) continue;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float s = r4d_wave_sum(acc[q][r]);
                if (!lane && r < nr) part[((r0 + r) * S + j) * chunks + chunk] = s;
            }
        }
    }
}

template <int HC>
__global__ __launch_bounds__(128) void mix_finish_kernel(const float* part, int chunks, MixOut o,
                                                         long rows) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32, CU = 32;
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const MixPre<HC> pr = mix_pre<HC>(lane, o);
    const bool vec = (chunks & 3) == 0 && ((uintptr_t)part & 15) == 0;
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int j = lane + 32 * q;
        const float* p = part + (row * (MIX + 1) + (j <= MIX ? j : 0)) * chunks;
        float s = 0.0f;
        if (vec) {  // CU chunks a batch as CU/4 16-byte loads, then their adds in chunk order
            for (int c = 0; c < chunks; c += CU) {
                float4 t[CU / 4];
#pragma unroll
                for (int u = 0; u < CU / 4; ++u)
                    t[u] = reinterpret_cast<const float4*>(p + c)[c + 4 * u < chunks ? u : 0];
#pragma unroll
                for (int u = 0; u < CU / 4; ++u) {
                    const float e[4] = { t[u].x, t[u].y, t[u].z, t[u].w };
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        if (c + 4 * u + k < chunks) s += e[k];
                }
            }
        } else {
            for (int c = 0; c < chunks; c += CU) {
                float t[CU];
#pragma unroll
                for (int u = 0; u < CU; ++u) t[u] = p[c + u < chunks ? c + u : c];
#pragma unroll
                for (int u = 0; u < CU; ++u)
                    if (c + u < chunks) s += t[u];
            }
        }
        d[q] = j <= MIX ? s : 0.0f;
    }
    mix_finish<HC>(d, lane, row, o, pr);
}

// Without scratch: one wave a row, the chunks in order, the same partials and the same sums.
template <int HC>
__global__ __launch_bounds__(128) void mix_fused_kernel(const uint16_t* h, long hp, const float* fn,
                                                        long fp, int chunks, MixOut o, long rows) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32;
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const MixPre<HC> pr = mix_pre<HC>(lane, o);
    float tot[MIX + 1];
#pragma unroll
    for (int j = 0; j <= MIX; ++j) tot[j] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        float acc[MIX + 1][1];
        mix_chunk<HC, 1>(h, hp, fn, fp, o.width, row, 1, c, lane, acc);
#pragma unroll
        for (int j = 0; j <= MIX; ++j) tot[j] += r4d_wave_sum(acc[j][0]);
    }
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        d[q] = 0.0f;
#pragma unroll
        for (int j = 0; j <= MIX; ++j)
            if (j == lane + 32 * q) d[q] = tot[j];
    }
    mix_finish<HC>(d, lane, row, o, pr);
}

long mix_chunks(long width) { return (width + kMixCols - 1) / kMixCols; }


template <int HC>
int mix_emit(RadStream stream, const RadTensor& h, const RadTensor& fn, const MixOut& o, long rows,
             void* scratch, int64_t scratch_bytes) {
    constexpr int MIX = HC * (HC + 2);
    const long chunks = mix_chunks(o.width);
    const int64_t need = (int64_t)rows * chunks * (MIX + 1) * (int64_t)sizeof(float);
    const unsigned fin = (unsigned)((rows + 3) / 4);
    if (scratch && scratch_bytes >= need) {
        float* part = (float*)scratch;
        // Rows a wave keeps (its sums' registers), and the fastest split per height on gfx1201
        // (n 5120, hc 4): one row up to 8, four rows 4 steps up to 24, then eight.
        constexpr int RB = HC <= 4 ? 8 : HC <= 6 ? 4 : 2, RM = RB < 4 ? RB : 4;
#define MIX_SPLIT(RS, TS)                                                                          \
    R4D_EMIT(stream, (mix_split_kernel<HC, RS, 8, TS>),                                            \
             dim3((unsigned)chunks, grid_rows((rows + (RS) - 1) / (RS))), dim3(256), 0,            \
             (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,                    \
             (long)fn.stride[0], part, o.width, (int)chunks, rows)
        const int rc = rows <= 8 ? MIX_SPLIT(1, kMixSteps)
                       : rows <= 24 ? MIX_SPLIT(RM, 4)
                                    : MIX_SPLIT(RB, 4);
#undef MIX_SPLIT
        if (rc < 0) return rc;
        return R4D_EMIT(stream, mix_finish_kernel<HC>, dim3(fin), dim3(128), 0, (const float*)part,
                        (int)chunks, o, rows);
    }
    return R4D_EMIT(stream, mix_fused_kernel<HC>, dim3(fin), dim3(128), 0, (const uint16_t*)h.data,
                    (long)h.stride[0], (const float*)fn.data, (long)fn.stride[0], (int)chunks, o,
                    rows);
}

bool mix_params(const RadArgs* a, long* n, long* hc, long* iters, float* eps, float* heps) {
    *n = (long)r4d_geti_or(a, "n", 0);
    *hc = (long)r4d_geti_or(a, "hc", 0);
    *iters = (long)r4d_geti_or(a, "iterations", 0);
    *eps = (float)r4d_getf(a, "eps", 0);
    *heps = (float)r4d_getf(a, "hc_eps", 0);
    return *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 &&
           *hc * *n <= (1L << 30) && std::isfinite(*eps) && std::isfinite(*heps) && *eps > 0 &&
           *heps >= 0;
}

int mhc_mix_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    long n, hc, iters;
    float eps, heps;
    if (!mix_params(a, &n, &hc, &iters, &eps, &heps)) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &fn = a->t[1], &sc = a->t[2], &bs = a->t[3];
    const RadTensor &pre = a->t[4], &post = a->t[5], &comb = a->t[6];
    const long width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || fn.dtype != RAD_F32 || sc.dtype != RAD_F32 || bs.dtype != RAD_F32 ||
        pre.dtype != RAD_F32 || post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, width) || !mat(fn, RAD_F32, mix, width) || !sc.data ||
        sc.rank != 1 || sc.shape[0] != 3 || !bs.data || bs.rank != 1 || bs.shape[0] != mix ||
        !mat_s(pre, RAD_F32, rows, hc) || !mat_s(post, RAD_F32, rows, hc) ||
        !mat_s(comb, RAD_F32, rows, hc * hc))
        return R4D_NO(RAD_E_SHAPE);
    MixOut o;
    o.scale = (const float*)sc.data;
    o.sst = (long)sc.stride[0];
    o.base = (const float*)bs.data;
    o.bst = (long)bs.stride[0];
    o.pre = (float*)pre.data;
    o.prp = (long)pre.stride[0];
    o.prs = (long)pre.stride[1];
    o.post = (float*)post.data;
    o.pop = (long)post.stride[0];
    o.pos = (long)post.stride[1];
    o.comb = (float*)comb.data;
    o.cp = (long)comb.stride[0];
    o.cs = (long)comb.stride[1];
    o.width = (int)width;
    o.iters = (int)iters;
    o.eps = eps;
    o.heps = heps;
    switch (hc) {
        case 2: return mix_emit<2>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 3: return mix_emit<3>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 4: return mix_emit<4>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 5: return mix_emit<5>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 6: return mix_emit<6>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 7: return mix_emit<7>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 8: return mix_emit<8>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        default: return R4D_NO(RAD_E_SHAPE);
    }
}

/* ==================================================================== mhc_collapse
 * One column a thread (eight with 16-byte rows), the streams summed from zero in order. */

__global__ __launch_bounds__(256) void collapse_kernel(const uint16_t* h, long hp, const float* pre,
                                                       long pp, long ps, uint16_t* x, long xp,
                                                       int n, int hc, long rows) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[r * pp + j * ps] * ds_bf(h[r * hp + (long)j * n + i]);
        x[r * xp + i] = ds_bf16(v);
    }
}

__global__ __launch_bounds__(256) void collapse_vec_kernel(const uint16_t* h, long hp,
                                                           const float* pre, long pp, long ps,
                                                           uint16_t* x, long xp, int n, int hc,
                                                           long rows) {
    const int i = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < hc; ++j) {
            const float pj = pre[r * pp + j * ps];
            const uint4 q = *reinterpret_cast<const uint4*>(h + r * hp + (long)j * n + i);
            const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += pj * ds_bf((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
        }
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[e] = (uint32_t)ds_bf16(v[2 * e]) | ((uint32_t)ds_bf16(v[2 * e + 1]) << 16);
        *reinterpret_cast<uint4*>(x + r * xp + i) = uint4{ o[0], o[1], o[2], o[3] };
    }
}

int mhc_collapse_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8 || hc * n > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &pre = a->t[1], &x = a->t[2];
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    const auto al16 = [](const RadTensor& t) {
        return ((uintptr_t)t.data & 15) == 0 && t.stride[0] % 8 == 0;
    };
    if (n % 8 == 0 && al16(h) && al16(x))
        return R4D_EMIT(stream, collapse_vec_kernel,
                        dim3((unsigned)((n / 8 + 255) / 256), grid_rows(rows)), dim3(256), 0,
                        (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                        (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data,
                        (long)x.stride[0], (int)n, (int)hc, rows);
    return R4D_EMIT(stream, collapse_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                    (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data, (long)x.stride[0],
                    (int)n, (int)hc, rows);
}

/* ==================================================================== softmax_pool
 * One thread a (entry, channel): the max, the exp sum and the weighted sum over the entry's rows,
 * each in row order; the exp is recomputed for the third pass (the same value). */

__global__ __launch_bounds__(256) void softmax_pool_kernel(const float* st, long sp, uint16_t* out,
                                                           long op, const int32_t* page, long pgs,
                                                           int n, int ratio, long W) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        if (page && page[w * pgs] < 0) continue;
        const float* b = st + w * ratio * sp;
        float mx = -INFINITY;
        for (int i = 0; i < ratio; ++i) mx = ds_max(mx, b[i * sp + n + c]);
        float den = 0;
        for (int i = 0; i < ratio; ++i) den += ds_exp(b[i * sp + n + c] - mx);
        float acc = 0;
        for (int i = 0; i < ratio; ++i) acc += ds_exp(b[i * sp + n + c] - mx) / den * b[i * sp + c];
        out[w * op + c] = ds_bf16(acc);
    }
}

int softmax_pool_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const long n = (long)r4d_geti_or(a, "n", 0), ratio = (long)r4d_geti_or(a, "ratio", 0);
    if (n < 1 || n > INT_MAX / 2 || ratio < 1 || ratio > 64 || st.rank != 2 || out.rank != 2 ||
        out.shape[0] < 1 || out.shape[1] != n || st.shape[0] != out.shape[0] * ratio ||
        st.shape[1] != 2 * n || (page && rad_tensor_numel(page) < out.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (st.dtype != RAD_F32 || out.dtype != RAD_BF16 || (page && page->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long W = out.shape[0];
    long pgs = 0;
    if (!mat(st, RAD_F32, W * ratio, 2 * n) || !mat(out, RAD_BF16, W, n) ||
        (page && !lin_step(*page, &pgs)))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, softmax_pool_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const float*)st.data, (long)st.stride[0], (uint16_t*)out.data,
                    (long)out.stride[0], page ? (const int32_t*)page->data : nullptr, pgs, (int)n,
                    (int)ratio, W);
}

/* ==================================================================== entry_store
 * One thread an element; a block's n values land at pg * n as libref writes them. */

__global__ __launch_bounds__(256) void entry_store_kernel(const uint16_t* x, long xp,
                                                          const int32_t* page, long pgs,
                                                          uint16_t* cache, long blocks, int n,
                                                          long W) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;
        cache[pg * n + i] = ds_bf16(ds_bf(x[w * xp + i]));
    }
}

/* format e4m3_e4m3_16 / e4m3_ue8m0_32: one thread a block of an entry packs it with
 * rad_kv_fp8.h's own code (libref's bytes); the entry's last thread zeroes its padding. */
__global__ __launch_bounds__(256) void entry_store_fp8_kernel(const uint16_t* x, long xp,
                                                              const int32_t* page, long pgs,
                                                              uint8_t* cache, long blocks, int n,
                                                              long ne, int f, long W) {
    const int nb = (int)rad_kvf_block(f), nblk = n / nb;
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= nblk) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;
        float v[32];
        const uint16_t* xr = x + w * xp + (long)b * nb;
        for (int i = 0; i < nb; ++i) v[i] = ds_bf(xr[i]);
        uint8_t c[32];
        uint8_t* e = cache + pg * ne;
        e[n + b] = rad_kvf_pack_block(f, v, c);
        for (int i = 0; i < nb; ++i) e[(long)b * nb + i] = c[i];
        if (b == nblk - 1)
            for (long i = n + nblk; i < ne; ++i) e[i] = 0;
    }
}

int entry_store_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    const int f = rad_kvf_parse(r4d_gets(a, "format"));
    if (f < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const long ne = f ? (long)rad_kvf_bytes(f, n) : n;
    if (n < 1 || n > INT_MAX || ne < 1 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        rad_tensor_numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != ne)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || page.dtype != RAD_I32 || cache.dtype != (f ? RAD_F8E4M3 : RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    const long W = x.shape[0];
    long pgs;
    if (!mat(x, RAD_BF16, W, n) || !lin_step(page, &pgs)) return R4D_NO(RAD_E_SHAPE);
    if (f) {
        if (!rad_tensor_is_contiguous(&cache)) return R4D_NO(RAD_E_SHAPE);
        const long nblk = n / rad_kvf_block(f);
        return R4D_EMIT(stream, entry_store_fp8_kernel, dim3((unsigned)((nblk + 255) / 256), grid_rows(W)),
                        dim3(256), 0, (const uint16_t*)x.data, (long)x.stride[0],
                        (const int32_t*)page.data, pgs, (uint8_t*)cache.data, (long)cache.shape[0],
                        (int)n, ne, f, W);
    }
    return R4D_EMIT(stream, entry_store_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const uint16_t*)x.data, (long)x.stride[0],
                    (const int32_t*)page.data, pgs, (uint16_t*)cache.data, (long)cache.shape[0],
                    (int)n, W);
}

/* ==================================================================== router_scored_topk
 * One wave a row, as r4d_glm.hip's router_sigmoid_topk: lane l holds experts l, l+32, ...; the
 * score is libref's (sigmoid, sqrt(softplus) or the softmax with its sum in expert order), round j
 * takes the largest (rank, ~expert) key below round j-1's and parks it on lane j, so equal ranks go
 * to the lower expert as libref's strict `>` scan takes them. A NaN or -inf rank is never taken.
 * The weights' sum runs in round order. */

enum { kSigmoid = 0, kSqrtSoftplus = 1, kSoftmax = 2 };

__device__ __forceinline__ unsigned long long ds_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank > -INFINITY ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long ds_wave_max_u64(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}

template <int OWN, int SCORE>
__global__ __launch_bounds__(128) void router_scored_kernel(const float* logits, long lp,
                                                            const float* bias, long bp, int32_t* ids,
                                                            long ip, float* weights, long wp,
                                                            long m, int ne, int k, int norm,
                                                            float factor) {
    const int lane = threadIdx.x & 31;
    const long r = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    float p[OWN], bv[OWN];  // every load first (a clamped index past ne, its value dropped)
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, ec = e < ne ? e : 0;
        const float x = logits[r * lp + ec];
        bv[i] = bias[(long)ec * bp];
        p[i] = e < ne ? x : 0.0f;
    }
    if (SCORE == kSoftmax) {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            if (lane + 32 * i < ne) mx = fmaxf(mx, p[i]);
        mx = r4d_wave_max(mx);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = lane + 32 * i < ne ? ds_exp(p[i] - mx) : 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            for (int l = 0; l < 32; ++l)
                if (32 * i + l < ne)
                    den += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p[i]), l));
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = p[i] / den;
    } else if (SCORE == kSigmoid) {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = -p[i];
        ds_exp_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = 1.0f / (1.0f + t[i]);
    } else {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = p[i];
        ds_exp_n(t);
        ds_log1p_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = sqrtf(p[i] > 20.0f ? p[i] : t[i]);
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne ? ds_rank_key(p[i] + bv[i], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = ds_wave_max_u64(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[r * ip + lane] = e;
        weights[r * wp + lane] = chosen * mult;
    }
}

template <int SCORE>
int router_emit(RadStream stream, const RadTensor& lg, const RadTensor& bias, const RadTensor& ids,
                const RadTensor& w, int ne, int k, int norm, float factor) {
    const long m = lg.shape[0];
    const dim3 grid((unsigned)((m + 3) / 4));
#define ROUTER(OWN)                                                                                \
    return R4D_EMIT(stream, (router_scored_kernel<OWN, SCORE>), grid, dim3(128), 0,                \
                    (const float*)lg.data, (long)lg.stride[0], (const float*)bias.data,            \
                    (long)bias.stride[0], (int32_t*)ids.data, (long)ids.stride[0],                 \
                    (float*)w.data, (long)w.stride[0], m, ne, k, norm, factor)
    if (ne <= 128) ROUTER(4);
    if (ne <= 256) ROUTER(8);
    if (ne <= 384) ROUTER(12);
    ROUTER(16);
#undef ROUTER
}

int router_scored_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc) return R4D_NO(RAD_E_UNSUPPORTED);
    int score;
    if (!std::strcmp(sc, "sigmoid")) score = kSigmoid;
    else if (!std::strcmp(sc, "sqrtsoftplus")) score = kSqrtSoftplus;
    else if (!std::strcmp(sc, "softmax")) score = kSoftmax;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne || ids.rank != 2 ||
        w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k)
        return R4D_NO(RAD_E_SHAPE);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    switch (score) {
        case kSigmoid:
            return router_emit<kSigmoid>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
        case kSqrtSoftplus:
            return router_emit<kSqrtSoftplus>(stream, lg, bias, ids, w, (int)ne, (int)k, norm,
                                              factor);
        default:
            return router_emit<kSoftmax>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
    }
}

/* ==================================================================== router_scored_topk_vl
 * router_scored_topk (above) with HF's per-row bias swap: a row whose `mask` is nonzero selects by
 * bias_vl, others by bias. The kernel and its rounding are the same; only the loaded bias plane
 * changes per row, so the ranks and weights stay libref's. */

template <int OWN, int SCORE>
__global__ __launch_bounds__(128) void router_scored_vl_kernel(
    const float* logits, long lp, const float* bias, long bp, const float* bias_vl, long bpv,
    const int32_t* mask, long mp, int32_t* ids, long ip, float* weights, long wp, long m, int ne,
    int k, int norm, float factor) {
    const int lane = threadIdx.x & 31;
    const long r = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    const float* bsel = mask[r * mp] != 0 ? bias_vl : bias;
    const long bs = mask[r * mp] != 0 ? bpv : bp;
    float p[OWN], bv[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, ec = e < ne ? e : 0;
        const float x = logits[r * lp + ec];
        bv[i] = bsel[(long)ec * bs];
        p[i] = e < ne ? x : 0.0f;
    }
    if (SCORE == kSoftmax) {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            if (lane + 32 * i < ne) mx = fmaxf(mx, p[i]);
        mx = r4d_wave_max(mx);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = lane + 32 * i < ne ? ds_exp(p[i] - mx) : 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            for (int l = 0; l < 32; ++l)
                if (32 * i + l < ne)
                    den += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p[i]), l));
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = p[i] / den;
    } else if (SCORE == kSigmoid) {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = -p[i];
        ds_exp_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = 1.0f / (1.0f + t[i]);
    } else {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = p[i];
        ds_exp_n(t);
        ds_log1p_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = sqrtf(p[i] > 20.0f ? p[i] : t[i]);
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne ? ds_rank_key(p[i] + bv[i], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = ds_wave_max_u64(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[r * ip + lane] = e;
        weights[r * wp + lane] = chosen * mult;
    }
}

template <int SCORE>
int router_vl_emit(RadStream stream, const RadTensor& lg, const RadTensor& bias,
                   const RadTensor& vl, const RadTensor& mask, const RadTensor& ids,
                   const RadTensor& w, int ne, int k, int norm, float factor) {
    const long m = lg.shape[0];
    const dim3 grid((unsigned)((m + 3) / 4));
#define ROUTER(OWN)                                                                                \
    return R4D_EMIT(stream, (router_scored_vl_kernel<OWN, SCORE>), grid, dim3(128), 0,             \
                    (const float*)lg.data, (long)lg.stride[0], (const float*)bias.data,            \
                    (long)bias.stride[0], (const float*)vl.data, (long)vl.stride[0],               \
                    (const int32_t*)mask.data, (long)mask.stride[0], (int32_t*)ids.data,           \
                    (long)ids.stride[0], (float*)w.data, (long)w.stride[0], m, ne, k, norm, factor)
    if (ne <= 128) ROUTER(4);
    if (ne <= 256) ROUTER(8);
    if (ne <= 384) ROUTER(12);
    ROUTER(16);
#undef ROUTER
}

int router_scored_vl_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 6) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc) return R4D_NO(RAD_E_UNSUPPORTED);
    int score;
    if (!std::strcmp(sc, "sigmoid")) score = kSigmoid;
    else if (!std::strcmp(sc, "sqrtsoftplus")) score = kSqrtSoftplus;
    else if (!std::strcmp(sc, "softmax")) score = kSoftmax;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &vl = a->t[2], &mask = a->t[3], &ids = a->t[4],
                    &w = a->t[5];
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !vl.data || !mask.data || !ids.data || !w.data || lg.rank != 2 ||
        lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        vl.rank != 1 || vl.shape[0] != ne || mask.rank != 1 || mask.shape[0] < lg.shape[0] ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k)
        return R4D_NO(RAD_E_SHAPE);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || vl.dtype != RAD_F32 ||
        ids.dtype != RAD_I32 || w.dtype != RAD_F32 || mask.dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    if (lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    switch (score) {
        case kSigmoid:
            return router_vl_emit<kSigmoid>(stream, lg, bias, vl, mask, ids, w, (int)ne, (int)k, norm,
                                            factor);
        case kSqrtSoftplus:
            return router_vl_emit<kSqrtSoftplus>(stream, lg, bias, vl, mask, ids, w, (int)ne, (int)k,
                                                 norm, factor);
        default:
            return router_vl_emit<kSoftmax>(stream, lg, bias, vl, mask, ids, w, (int)ne, (int)k, norm,
                                            factor);
    }
}

__global__ __launch_bounds__(256) void embed_planes_kernel(
    const int32_t* tok, long ts, const uint8_t* codes, long rb, uint32_t cdt, const void* scale,
    uint32_t sdt, long sp, long ss, const void* table, uint32_t tdt, uint16_t* x, long xp, int E,
    int G, long V, long voff, float mult, long M) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= E) return;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long id = (long)tok[m * ts] - voff;
        float val = 0.0f;
        if (id >= 0 && id < V) {
            const uint8_t* row = codes + id * rb;
            int code;
            if (cdt == RAD_I8) {
                code = (int)(int8_t)row[i];
            } else if (cdt == RAD_U8) {
                code = (int)row[i];
            } else {
                const int u = (row[i >> 1] >> ((i & 1) * 4)) & 15;
                code = cdt == RAD_I4 ? (u ^ 8) - 8 : u;
            }
            const float v = table ? ds_ld(tdt, table, code) : (float)code;
            val = v * ds_ld(sdt, scale, id * sp + (long)(i / G) * ss) * mult;
        }
        x[m * xp + i] = ds_bf16(val);
    }
}

bool float_plane_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_E8M0;
}
bool table_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_I8 ||
           dt == RAD_U8;
}

int embed_planes_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &cd = a->t[1], &sc = a->t[2], &tb = a->t[3], &x = a->t[4];
    const bool tab = tb.data != nullptr;
    const long E = (long)r4d_geti_or(a, "n_embd", 0), V = (long)r4d_geti_or(a, "n_vocab", 0);
    const long G = (long)r4d_geti_or(a, "group", 0), voff = (long)r4d_geti_or(a, "vocab_offset", 0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    if (E <= 0 || E > INT_MAX || V <= 0 || G <= 0 || E % G || voff < 0 || cd.rank != 2 ||
        cd.shape[0] < V || cd.shape[1] != E || sc.rank != 2 || sc.shape[0] < V ||
        sc.shape[1] != E / G)
        return R4D_NO(RAD_E_SHAPE);
    const uint32_t cdt = cd.dtype;
    const int bits = (cdt == RAD_U4 || cdt == RAD_I4) ? 4 : (cdt == RAD_U8 || cdt == RAD_I8) ? 8 : 0;
    if (!bits) return R4D_NO(RAD_E_DTYPE);
    if (tab && (cdt == RAD_I4 || cdt == RAD_I8 || rad_tensor_numel(&tb) < (int64_t)1 << bits))
        return R4D_NO(RAD_E_DTYPE);
    if (!float_plane_dt(sc.dtype) || (tab && !table_dt(tb.dtype)) || tok.dtype != RAD_I32 ||
        x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long rb = (long)rad_dtype_bytes(cdt, E);
    const long nx = (long)rad_tensor_numel(&x);
    if (rb <= 0 || nx < E || nx % E) return R4D_NO(RAD_E_SHAPE);
    const long M = nx / E;
    long xp;
    if (x.rank == 2 && x.shape[1] == E) {
        if (x.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        xp = (long)x.stride[0];
    } else if (rad_tensor_is_contiguous(&x)) {
        xp = E;
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    long ts;
    if (rad_tensor_numel(&tok) < M || !lin_step(tok, &ts)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, embed_planes_kernel, dim3((unsigned)((E + 255) / 256), grid_rows(M)),
                    dim3(256), 0, (const int32_t*)tok.data, ts, (const uint8_t*)cd.data, rb, cdt,
                    (const void*)sc.data, sc.dtype, (long)sc.stride[0], (long)sc.stride[1],
                    tab ? (const void*)tb.data : nullptr, tab ? tb.dtype : 0u, (uint16_t*)x.data,
                    xp, (int)E, (int)G, V, voff, mult, M);
}

/* ==================================================================== fused chains
 * rmsnorm_rope_roundtrip  rmsnorm -> rope_rows -> quant_roundtrip   (a window or index key)
 * rope_rows_roundtrip     rope_rows -> quant_roundtrip              (attention output, keys)
 * mhc_collapse_rmsnorm    mhc_collapse -> rmsnorm                   (a sublayer's input)
 * mhc_post_collapse_rmsnorm  mhc_post -> mhc_collapse -> rmsnorm    (a sublayer's output into the
 *                         next sublayer's input)
 * One launch each where a decode step issued two or three dependent ones. Byte-identical to the
 * chain of rows (r4d_fuse.cpp declares it, rad-kbench's fusion oracle runs it): every step forms
 * its values as its own row does and rounds where that row stores. The norm is rmsnorm_bf16's
 * (r4d_model_bf16.hip rmsnorm_body): r4d_act_threads(M, n) threads a row, column i on thread
 * i % width, the same block_sum, rsqrtf, product order and f32_to_bf16 (no NaN quieting there);
 * rope_rows' pair expressions and quant_roundtrip's lane-group amax are the rows above. */

constexpr uint32_t kFzWave = 32;
// Columns a thread loads ahead per batch in the fused norms (rows of up to kFzCols * width stay in registers).
constexpr int kFzCols = 4;

// r4d_model_bf16.hip's block_sum: the wave butterfly, then the waves' sums in order from zero.
template <uint32_t NW>
__device__ __forceinline__ float fz_block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kFzWave - 1), wave = threadIdx.x / kFzWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// rope_rows' operands and geometry, as rope_rows_body passes them to rope_rows_kernel.
struct FzRope {
    const float* cs;
    long cp, cst;
    const int32_t* pos;
    long ps, tab_rows;
    int hd, half, off, pairs, gptj;
    float sign;
};

// quant_group_kernel's step over a row held in `buf`: lane groups of B columns, `width` threads.
template <int B>
__device__ __forceinline__ void fz_quant_group(const uint16_t* buf, uint16_t* yr, int n,
                                               uint32_t width, int fmt, int sc) {
    for (int base = 0; base < n; base += (int)width) {
        const int col = base + (int)threadIdx.x;
        const bool in = col < n;
        const float v = in ? ds_bf(buf[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        if (in) yr[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

// quant_serial_kernel's step: one thread a block.
__device__ __forceinline__ void fz_quant_serial(const uint16_t* buf, uint16_t* yr, int n, int block,
                                                uint32_t width, int fmt, int sc) {
    for (int b = (int)threadIdx.x; b < n / block; b += (int)width) {
        const uint16_t* p = buf + (long)b * block;
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = ds_max(amax, fabsf(ds_bf(p[i])));
        const float s = block_scale(amax, fmt, sc);
        for (int i = 0; i < block; ++i) yr[(long)b * block + i] = ds_bf16(roundtrip(ds_bf(p[i]), s, fmt));
    }
}

/* rmsnorm_rope_roundtrip: one workgroup a row (rmsnorm's grid and width); the row passes between
 * the steps through LDS as the bf16 each step stores. */
template <bool WF32, uint32_t THR>
__global__ __launch_bounds__(THR) void norm_rope_rt_kernel(
    const uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint32_t y_ld, uint32_t n, float eps, float wadd, float mult,
    uint32_t M, FzRope r, int block, int fmt, int sc) {
    extern __shared__ uint16_t fz_row[];
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;

    // rmsnorm_body's sums; the row and its gains are loaded a batch of kFzCols columns ahead, and
    // stay in registers across the barrier when the row is one batch (n <= kFzCols * THR).
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            xv[c] = bf16_to_f32(xb[c]);
            if (i0 + (uint32_t)c * THR < n) ss += xv[c] * xv[c];
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) fz_row[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float g = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
            fz_row[i] = f32_to_bf16(bf16_to_f32(xr[i]) * s * (g + wadd) * mult);
        }
    }
    __syncthreads();

    const long p = r.pos[(long)m * r.ps];
    if (p >= 0 && p < r.tab_rows)
        for (int q = (int)threadIdx.x; q < r.pairs; q += (int)THR) {
            const int hh = q / r.half, i = q - hh * r.half;
            const long ia = (long)hh * r.hd + r.off + (r.gptj ? 2 * i : i);
            const long ib = r.gptj ? ia + 1 : ia + r.half;
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = ds_bf(fz_row[ia]), vb = ds_bf(fz_row[ib]);
            fz_row[ia] = ds_bf16(va * c - vb * sn);
            fz_row[ib] = ds_bf16(va * sn + vb * c);
        }
    __syncthreads();

    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    switch (block) {
        case 1: fz_quant_group<1>(fz_row, yr, (int)n, THR, fmt, sc); break;
        case 2: fz_quant_group<2>(fz_row, yr, (int)n, THR, fmt, sc); break;
        case 4: fz_quant_group<4>(fz_row, yr, (int)n, THR, fmt, sc); break;
        case 8: fz_quant_group<8>(fz_row, yr, (int)n, THR, fmt, sc); break;
        case 16: fz_quant_group<16>(fz_row, yr, (int)n, THR, fmt, sc); break;
        case 32: fz_quant_group<32>(fz_row, yr, (int)n, THR, fmt, sc); break;
        default: fz_quant_serial(fz_row, yr, (int)n, block, THR, fmt, sc); break;
    }
}

/* rope_rows_roundtrip, in place: quant_group_kernel's grid (a thread a column, rows over grid.y);
 * a rotated column takes its gptj partner (the other lane of its even/odd pair, the same wave)
 * by a shuffle, forms its half of rope_rows' pair and rounds it as rope_rows stores it. */
template <int B>
__global__ __launch_bounds__(256) void rope_rt_kernel(uint16_t* x, long xp, int n, long rows,
                                                      FzRope r, int fmt, int sc) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    const int hh = col / r.hd, d = col - hh * r.hd - r.off;
    const bool rot = in && d >= 0 && d < 2 * r.half;
    const int i = d >> 1;
    const bool is_a = (d & 1) == 0;
    for (long m = blockIdx.y; m < rows; m += gridDim.y) {
        uint16_t* row = x + m * xp;
        const uint32_t h = in ? (uint32_t)row[col] : 0u;
        const uint32_t hp = (uint32_t)__shfl_xor((int)h, 1);
        float v = ds_bf((uint16_t)h);
        const long p = r.pos[m * r.ps];
        if (rot && p >= 0 && p < r.tab_rows) {
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = is_a ? v : ds_bf((uint16_t)hp), vb = is_a ? ds_bf((uint16_t)hp) : v;
            v = ds_bf(is_a ? ds_bf16(va * c - vb * sn) : ds_bf16(va * sn + vb * c));
        }
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        if (in) row[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

/* mhc_collapse_rmsnorm: rmsnorm's grid and width. A thread owns columns i = tid + k*THR (the norm's
 * own split), collapses each (the streams from zero in order, collapse_kernel's sum), stores it and
 * folds the stored bf16 into the sum of squares in that order. Loads go out a batch of kFzCols
 * columns at a time (every stream and the gains, at clamped addresses) ahead of the arithmetic; when
 * the row is one batch (n <= kFzCols * THR: DeepSeek-V4.1's decode rows) the collapsed values and
 * gains stay in registers across the barrier, else the second pass re-reads what this thread stored
 * (as add_rmsnorm_body), so no barrier between. HCM: streams loaded a column (4 or 8, hc <= HCM). */
template <bool WF32, uint32_t THR, int HCM>
__global__ __launch_bounds__(THR) void collapse_norm_kernel(
    const uint16_t* __restrict__ h, long hp, const float* __restrict__ pre, long pp, long pst,
    uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t y_ld, uint32_t n, int hc, float eps, float wadd, float mult, uint32_t M) {
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* __restrict__ hr = h + (long)m * hp;
    const float* __restrict__ pr = pre + (long)m * pp;
    uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    float pj[HCM];
#pragma unroll
    for (int j = 0; j < HCM; ++j) pj[j] = pr[(long)(j < hc ? j : hc - 1) * pst];
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint32_t hv[kFzCols][HCM];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
#pragma unroll
            for (int j = 0; j < HCM; ++j) hv[c][j] = hr[(long)(j < hc ? j : hc - 1) * n + i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            float v = 0;
#pragma unroll
            for (int j = 0; j < HCM; ++j)
                if (j < hc) v += pj[j] * ds_bf((uint16_t)hv[c][j]);
            const uint16_t b = ds_bf16(v);
            xv[c] = bf16_to_f32(b);
            if (i < n) {
                xr[i] = b;
                ss += xv[c] * xv[c];
            }
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
        return;
    }
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(bf16_to_f32(xb[c]) * s * (gv[c] + wadd) * mult);
        }
    }
}

/* mhc_post_collapse_rmsnorm: collapse_norm_kernel with mhc_post in front, column-local. For each of
 * its columns a thread forms every output stream k as mhc_post_kernel does (from zero, comb[j][k] *
 * h[j] over the source streams in order, then + post[k] * x, f32_to_bf16), stores it, and collapses
 * the stored values (collapse_kernel's sum over k, ds_bf16); the norm is collapse_norm_kernel's. */
template <bool WF32, uint32_t THR, int HCM>
__global__ __launch_bounds__(THR) void post_collapse_norm_kernel(
    const uint16_t* __restrict__ xin, long xp, const uint16_t* __restrict__ h, long hp,
    const float* __restrict__ post, long ptp, const float* __restrict__ comb, long cbp,
    uint16_t* __restrict__ out, long op, const float* __restrict__ pre, long pp, long pst,
    uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t y_ld, uint32_t n, int hc, float eps, float wadd, float mult, uint32_t M) {
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* __restrict__ xir = xin + (long)m * xp;
    const uint16_t* __restrict__ hr = h + (long)m * hp;
    uint16_t* __restrict__ orow = out + (long)m * op;
    const float* __restrict__ pr = pre + (long)m * pp;
    uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    float pj[HCM], pk[HCM], cb[HCM][HCM];
#pragma unroll
    for (int k = 0; k < HCM; ++k) {
        const int kk = k < hc ? k : hc - 1;
        pj[k] = pr[(long)kk * pst];
        pk[k] = post[(long)m * ptp + kk];
#pragma unroll
        for (int j = 0; j < HCM; ++j) cb[j][k] = comb[(long)m * cbp + (j < hc ? j : hc - 1) * hc + kk];
    }
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint32_t hv[kFzCols][HCM], xi[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
#pragma unroll
            for (int j = 0; j < HCM; ++j) hv[c][j] = hr[(long)(j < hc ? j : hc - 1) * n + i];
            xi[c] = xir[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            float v = 0;
#pragma unroll
            for (int k = 0; k < HCM; ++k) {
                if (k >= hc) continue;
                float u = 0;
#pragma unroll
                for (int j = 0; j < HCM; ++j)
                    if (j < hc) u += cb[j][k] * bf16_to_f32((uint16_t)hv[c][j]);
                u += pk[k] * bf16_to_f32((uint16_t)xi[c]);
                const uint16_t o = f32_to_bf16(u);
                if (i < n) orow[(long)k * n + i] = o;
                v += pj[k] * ds_bf(o);
            }
            const uint16_t b = ds_bf16(v);
            xv[c] = bf16_to_f32(b);
            if (i < n) {
                xr[i] = b;
                ss += xv[c] * xv[c];
            }
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
        return;
    }
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(bf16_to_f32(xb[c]) * s * (gv[c] + wadd) * mult);
        }
    }
}

// quant_rt_body's format and scale names.
bool fz_quant_fmt(const RadArgs* a, int* fmt, int* sc) {
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "scale");
    if (!fs || !ss) return false;
    if (!std::strcmp(fs, "fp8_e4m3")) *fmt = kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) *fmt = kFp4;
    else return false;
    if (!std::strcmp(ss, "ue8m0")) *sc = kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && *fmt == kFp4) *sc = kE4m3;
    else if (!std::strcmp(ss, "f32") && *fmt == kFp8) *sc = kScaleF32;
    else return false;
    return true;
}

// rope_rows_body's checks over `x` [M, nh*hd] and its cos_sin / positions, into an FzRope.
int fz_rope_args(const RadArgs* a, const RadTensor& x, const RadTensor& cs, const RadTensor& pos,
                 long M, FzRope* r) {
    const long nh = (long)r4d_geti_or(a, "n_head", 0), hd = (long)r4d_geti_or(a, "head_dim", 0);
    const long rot = (long)r4d_geti_or(a, "rot", 0), off = (long)r4d_geti_or(a, "off", -1);
    const long inv = (long)r4d_geti_or(a, "inverse", -1);
    const char* mode = r4d_gets(a, "mode");
    if (!mode) return RAD_E_UNSUPPORTED;
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return RAD_E_UNSUPPORTED;
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1) || nh * hd > INT_MAX)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || cs.dtype != RAD_F32 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    long ps;
    if (!mat(x, RAD_BF16, M, nh * hd) || !cs.data || cs.rank != 2 || cs.shape[1] != rot ||
        rad_tensor_numel(&pos) != M || !lin_step(pos, &ps))
        return RAD_E_SHAPE;
    r->cs = (const float*)cs.data;
    r->cp = (long)cs.stride[0];
    r->cst = (long)cs.stride[1];
    r->pos = (const int32_t*)pos.data;
    r->ps = ps;
    r->tab_rows = (long)cs.shape[0];
    r->hd = (int)hd;
    r->half = (int)(rot / 2);
    r->off = (int)off;
    r->pairs = (int)(nh * (rot / 2));
    r->gptj = gptj ? 1 : 0;
    r->sign = inv ? -1.0f : 1.0f;
    return RAD_OK;
}

// Most columns the fused norm holds in LDS (the row as bf16); the row's constraint says the same.
constexpr long kFzMaxN = 16384;

/* x [M, n] (rows as rmsnorm reads them), w, y [M, n], cos_sin, positions. */
int norm_rope_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2], &cs = a->t[3], &pos = a->t[4];
    if (!x.data || !w.data || !y.data || !cs.data || !pos.data) return R4D_NO(RAD_E_INVAL);
    if (x.dtype != RAD_BF16 || y.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || n > kFzMaxN || r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(&x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || M > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    FzRope r;
    const int rc = fz_rope_args(a, y, cs, pos, (long)M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    if ((long long)r.hd * (r.pairs / r.half) != n) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)M, (int)n);
    const uint32_t lds = (uint32_t)(n * sizeof(uint16_t));
#define FZ_NRQ(WF, T)                                                                             \
    R4D_EMIT(stream, (norm_rope_rt_kernel<WF, T>), dim3(r4d_cliff_rows((uint32_t)M, 1, T / kFzWave)), \
             dim3(T), lds, (const uint16_t*)x.data, (uint32_t)x_ld, (const void*)w.data,          \
             (uint16_t*)y.data, (uint32_t)y.stride[0], (uint32_t)n, eps, wadd, mult, (uint32_t)M, \
             r, (int)block, fmt, sc)
#define FZ_NRQ_W(WF)                                                                              \
    return thr == 1024 ? FZ_NRQ(WF, 1024) : thr == 512 ? FZ_NRQ(WF, 512) : FZ_NRQ(WF, 256)
    if (w.dtype == RAD_F32) FZ_NRQ_W(true);
    FZ_NRQ_W(false);
#undef FZ_NRQ_W
#undef FZ_NRQ
}

/* x [M, n_head*head_dim] in place, cos_sin, positions. The lane-group form needs a gptj pair on
 * one wave's even/odd lanes (head_dim and off even, the row's constraints) and a power-of-two block
 * up to 32; any other geometry runs rope_rows then quant_roundtrip as two launches, the same bits. */
int rope_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = (long)x.shape[0];
    FzRope r;
    const int rc = fz_rope_args(a, x, cs, pos, M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    const long n = (long)r.hd * (r.pairs / r.half);
    const long block = (long)r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    const bool lanes = r.gptj && !(r.hd & 1) && !(r.off & 1) &&
                       (block == 1 || block == 2 || block == 4 || block == 8 || block == 16 ||
                        block == 32);
    if (!lanes) {
        const int rr = rope_rows_body(a, stream);
        if (rr != RAD_OK) return rr;
        RadParam pp[24];
        if (a->n_p > 23) return R4D_NO(RAD_E_INVAL);
        std::memcpy(pp, a->p, sizeof(RadParam) * (size_t)a->n_p);
        RadParam& pn = pp[a->n_p];
        std::memset(&pn, 0, sizeof pn);
        pn.key = "n";
        pn.kind = RAD_P_INT;
        pn.ival = n;
        RadArgs q = *a;
        q.p = pp;
        q.n_p = a->n_p + 1;
        return quant_rt_body(&q, stream);
    }
    const dim3 grid((unsigned)((n + 255) / 256), grid_rows(M));
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
#define FZ_RRT(B)                                                                                  \
    case B:                                                                                        \
        return R4D_EMIT(stream, rope_rt_kernel<B>, grid, dim3(256), 0, xd, xp, (int)n, M, r, fmt, sc)
    switch (block) {
        FZ_RRT(1);
        FZ_RRT(2);
        FZ_RRT(4);
        FZ_RRT(8);
        FZ_RRT(16);
        FZ_RRT(32);
        default: break;
    }
#undef FZ_RRT
    return R4D_NO(RAD_E_SHAPE);
}

/* h [M, hc*n], pre [M, hc] f32, x [M, n] (the collapsed row, stored), w, y (rows as rmsnorm). */
int collapse_norm_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8 || hc * n > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &pre = a->t[1], &x = a->t[2], &w = a->t[3], &y = a->t[4];
    if (!w.data || !y.data) return R4D_NO(RAD_E_INVAL);
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16 || y.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < rows || rows > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)rows, (int)n);
#define FZ_CN(WF, T, HM)                                                                           \
    R4D_EMIT(stream, (collapse_norm_kernel<WF, T, HM>),                                            \
             dim3(r4d_cliff_rows((uint32_t)rows, 1, T / kFzWave)), dim3(T), 0,                     \
             (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,                   \
             (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data, (uint32_t)x.stride[0],   \
             (const void*)w.data, (uint16_t*)y.data, (uint32_t)y_ld, (uint32_t)n, (int)hc, eps,    \
             wadd, mult, (uint32_t)rows)
#define FZ_CN_T(WF, HM)                                                                            \
    return thr == 1024 ? FZ_CN(WF, 1024, HM) : thr == 512 ? FZ_CN(WF, 512, HM) : FZ_CN(WF, 256, HM)
#define FZ_CN_W(WF)                                                                                \
    do { if (hc <= 4) FZ_CN_T(WF, 4); FZ_CN_T(WF, 8); } while (0)
    if (w.dtype == RAD_F32) FZ_CN_W(true);
    FZ_CN_W(false);
#undef FZ_CN_W
#undef FZ_CN_T
#undef FZ_CN
}

/* x [M, n], h [M, hc*n], post [M, hc], comb [M, hc*hc] f32 -> out [M, hc*n] (mhc_post_bf16's
 * checks: n <= 16384, out not h); pre, xc, w, y as mhc_collapse_rmsnorm. */
int post_collapse_norm_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || n > 16384 || hc < 2 || hc > 8) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &xi = a->t[0], &h = a->t[1], &po = a->t[2], &cb = a->t[3], &o = a->t[4];
    const RadTensor &pre = a->t[5], &x = a->t[6], &w = a->t[7], &y = a->t[8];
    if (!w.data || !y.data) return R4D_NO(RAD_E_INVAL);
    if (xi.rank != 2 || xi.shape[0] < 1 || xi.shape[0] > 65535) return R4D_NO(RAD_E_SHAPE);
    const long rows = xi.shape[0];
    if (xi.dtype != RAD_BF16 || h.dtype != RAD_BF16 || po.dtype != RAD_F32 || cb.dtype != RAD_F32 ||
        o.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16 || y.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(xi, RAD_BF16, rows, n) || !mat(h, RAD_BF16, rows, hc * n) ||
        !mat(po, RAD_F32, rows, hc) || !mat(cb, RAD_F32, rows, hc * hc) ||
        !mat(o, RAD_BF16, rows, hc * n) || o.data == h.data || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < rows) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)rows, (int)n);
#define FZ_PCN(WF, T, HM)                                                                          \
    R4D_EMIT(stream, (post_collapse_norm_kernel<WF, T, HM>),                                       \
             dim3(r4d_cliff_rows((uint32_t)rows, 1, T / kFzWave)), dim3(T), 0,                     \
             (const uint16_t*)xi.data, (long)xi.stride[0], (const uint16_t*)h.data,                \
             (long)h.stride[0], (const float*)po.data, (long)po.stride[0],                         \
             (const float*)cb.data, (long)cb.stride[0], (uint16_t*)o.data, (long)o.stride[0],      \
             (const float*)pre.data, (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data,  \
             (uint32_t)x.stride[0], (const void*)w.data, (uint16_t*)y.data, (uint32_t)y_ld,        \
             (uint32_t)n, (int)hc, eps, wadd, mult, (uint32_t)rows)
#define FZ_PCN_T(WF, HM)                                                                           \
    return thr == 1024 ? FZ_PCN(WF, 1024, HM) : thr == 512 ? FZ_PCN(WF, 512, HM) : FZ_PCN(WF, 256, HM)
#define FZ_PCN_W(WF)                                                                               \
    do { if (hc <= 4) FZ_PCN_T(WF, 4); FZ_PCN_T(WF, 8); } while (0)
    if (w.dtype == RAD_F32) FZ_PCN_W(true);
    FZ_PCN_W(false);
#undef FZ_PCN_W
#undef FZ_PCN_T
#undef FZ_PCN
}

/* ===================== ds41_vit_rope2d / ds41_pack3x3 (the tower's own geometry, ref_pixtral.cpp)
 * rope2d: one workgroup a row. Each column i of the rotary half carries (axis, j) -- axis = i /
 * (D/4) picks coord[0] or coord[1], and the angle is pos * theta^(-2j/(D/2)) -- so the two axes
 * ride one ladder as HF's apply_rotary does. Both q and k are rotated, v is untouched, and each
 * product narrows once. pack3x3: one workgroup an output row; the workgroup first walks the
 * segments' grids (h*w patches each, ceil(h/K) * ceil(w/K) rows each) to find its segment and
 * cell, then fills K*K*N values, zeros where the padded grid runs past the true grid. */

template <uint32_t DT>
__device__ __forceinline__ void ds_put(void* p, long i, float v) {
    if (DT == RAD_BF16) ((uint16_t*)p)[i] = ds_bf16(v);
    else ((float*)p)[i] = v;
}

template <uint32_t DT>
__global__ __launch_bounds__(64) void ds41_vit_rope2d_kernel(void* x, long xp, const int32_t* crd,
                                                             long ca, long cb, long m, int D, int H,
                                                             float theta) {
    const long p = blockIdx.x;
    if (p >= m) return;
    const int rot = D / 2, quad = D / 4;
    const int row = crd[0 * ca + p * cb], col = crd[1 * ca + p * cb];
    for (int h = 0; h < 2 * H; ++h)
        for (int i = threadIdx.x; i < rot; i += blockDim.x) {
            const int axis = i / quad, j = i % quad;
            const float pos = (float)(axis ? col : row);
            const float inv = 1.0f / __powf(theta, (float)(2 * j) / (float)rot);
            const float ang = pos * inv;
            const float c = cosf(ang), s = sinf(ang);
            const long u = p * xp + (long)(h * D + i), v = u + rot;
            const float a = ds_ld(DT, x, u), b = ds_ld(DT, x, v);
            ds_put<DT>(x, u, a * c - b * s);
            ds_put<DT>(x, v, b * c + a * s);
        }
}

template <uint32_t DT>
__global__ __launch_bounds__(256) void ds41_pack3x3_kernel(const void* x, long xp,
                                                            const int32_t* crd, long ca, long cb,
                                                            void* y, long yp, long M, long N, int K) {
    const long r = blockIdx.x;
    long p = 0, rr = 0, h = 0, w = 0, seg = 0;
    int oy = 0, ox = 0;
    while (p < M) {
        const long hh = crd[2 * ca + p * cb], ww = crd[3 * ca + p * cb];
        if (hh <= 0 || ww <= 0) break;
        const long oh = (hh + K - 1) / K, ow = (ww + K - 1) / K;
        if (r < rr + oh * ow) {
            seg = p;
            h = hh;
            w = ww;
            oy = (int)((r - rr) / ow);
            ox = (int)((r - rr) % ow);
            break;
        }
        rr += oh * ow;
        p += hh * ww;
    }
    const long nb = (long)K * K * (long)N;
    /* Column order is the checkpoint's (channel, kernel row, kernel column): one merged block's
     * row is N pixels' worth of K*K patches, pixel-major, matching HF's unfold of the
     * (channel, row, col) view -- not patch-major. */
    for (long j = threadIdx.x; j < nb; j += blockDim.x) {
        const long cell = j % (long)(K * K), jj = j / (long)(K * K);
        const long yy = (long)oy * K + cell / K, xx = (long)ox * K + cell % K;
        float v = 0.0f;
        if (yy < h && xx < w) v = ds_ld(DT, x, (seg + yy * w + xx) * xp + jj);
        ds_put<DT>(y, r * yp + j, v);
    }
}

int vit_rope2d_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &crd = a->t[1];
    /* The ranged M in the args is the band's upper bound, not this issue's rows. The view's
     * leading dimension is the count (brows / praw2). */
    const long m = x.rank == 2 ? (long)x.shape[0] : 0;
    const int D = (int)r4d_geti_or(a, "head_dim", 0), H = (int)r4d_geti_or(a, "n_head", 0);
    const float theta = (float)r4d_getf(a, "theta", 0);
    if (m < 1 || D < 4 || D % 4 || H < 1 || !std::isfinite(theta) || theta <= 0)
        return R4D_NO(RAD_E_SHAPE);
    if (x.rank != 2 || x.shape[0] < m || x.shape[1] != (long)3 * H * D || crd.rank != 2 ||
        crd.shape[0] < 2 || crd.shape[1] < m)
        return R4D_NO(RAD_E_SHAPE);
    if (crd.dtype != RAD_I32 || (x.dtype != RAD_BF16 && x.dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const dim3 grid((unsigned)m);
    if (x.dtype == RAD_BF16)
        return R4D_EMIT(stream, ds41_vit_rope2d_kernel<RAD_BF16>, grid, dim3(64), 0, x.data,
                        (long)x.stride[0], crd.data, (long)crd.stride[0], (long)crd.stride[1], m, D,
                        H, theta);
    return R4D_EMIT(stream, ds41_vit_rope2d_kernel<RAD_F32>, grid, dim3(64), 0, x.data,
                    (long)x.stride[0], crd.data, (long)crd.stride[0], (long)crd.stride[1], m, D, H,
                    theta);
}

/* ds41_span_rows: copy the three learned vectors into the typed-row table (row 0 start, 2
 * newline, 3 end; row 1 is the encoder's own rows and is not written). No arithmetic: one bf16
 * pair moves per element. */
__global__ __launch_bounds__(256) void ds41_span_rows_kernel(
    const uint16_t* __restrict__ s, const uint16_t* __restrict__ nl,
    const uint16_t* __restrict__ e, uint16_t* __restrict__ y, unsigned n, unsigned pitch) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        y[i] = s[i];
        y[2u * pitch + i] = nl[i];
        y[3u * pitch + i] = e[i];
    }
}

int span_rows_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[3].data)
        return R4D_NO(RAD_E_INVAL);
    const RadTensor &s = a->t[0], &nl = a->t[1], &e = a->t[2], &y = a->t[3];
    const long n = (long)r4d_geti_or(a, "n", 0);
    if (n < 1 || y.rank != 2 || y.shape[0] != 4 || y.shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (y.dtype != RAD_BF16 || s.dtype != RAD_BF16 || nl.dtype != RAD_BF16 || e.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (y.stride[1] != 1 || y.stride[0] <= 0) return R4D_NO(RAD_E_SHAPE);
    if (rad_tensor_numel(&s) < n || rad_tensor_numel(&nl) < n || rad_tensor_numel(&e) < n)
        return R4D_NO(RAD_E_SHAPE);
    const dim3 grid((unsigned)((n + 255) / 256));
    return R4D_EMIT(stream, ds41_span_rows_kernel, grid, dim3(256), 0, s.data, nl.data, e.data,
                    y.data, (unsigned)n, (unsigned)y.stride[0]);
}

int pack3x3_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &crd = a->t[1], &y = a->t[2];
    const long m = x.rank == 2 ? (long)x.shape[0] : 0, N = (long)r4d_geti_or(a, "n", 0);
    const int K = (int)r4d_geti_or(a, "merge", 0);
    if (m < 1 || N < 1 || K < 2 || x.rank != 2 || x.shape[0] != m || x.shape[1] != N ||
        crd.rank != 2 || crd.shape[0] != 4 || crd.shape[1] != m || y.rank != 2 ||
        y.shape[1] != N * K * K) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (crd.dtype != RAD_I32 || x.dtype != y.dtype || (x.dtype != RAD_BF16 && x.dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (y.stride[1] != 1 || x.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    /* The output row count is the issue's own view row count: the engine computes `rows` from the
     * segment grids (the same h and w values the coordinates carry) and passes it as y.shape[0].
     * Do NOT walk crd here. The host wrapper runs while the asynchronous coordinate upload may
     * still be in flight, so reading a raw device pointer returns the previous pass's
     * bytes (or zeros), and a computed `rows` of 0 aborts a valid step and stops the engine.
     * The device kernel walks crd at execution time, after the copy lands. */
    const long rows = (long)y.shape[0];
    if (rows < 1 || rows > m) return R4D_NO(RAD_E_SHAPE);
    const dim3 grid((unsigned)rows);
    if (x.dtype == RAD_BF16)
        return R4D_EMIT(stream, ds41_pack3x3_kernel<RAD_BF16>, grid, dim3(256), 0, x.data,
                        (long)x.stride[0], crd.data, (long)crd.stride[0], (long)crd.stride[1],
                        y.data, (long)y.stride[0], m, N, K);
    return R4D_EMIT(stream, ds41_pack3x3_kernel<RAD_F32>, grid, dim3(256), 0, x.data,
                    (long)x.stride[0], crd.data, (long)crd.stride[0], (long)crd.stride[1], y.data,
                    (long)y.stride[0], m, N, K);
}

}  // namespace

extern "C" int r4d_rad_token_map(const RadArgs* a, RadStream s) { return token_map_body(a, s); }
extern "C" int r4d_rad_stream_gate_add(const RadArgs* a, RadStream s) {
    return stream_gate_body(a, s);
}
extern "C" int r4d_rad_ds41_rope_rows(const RadArgs* a, RadStream s) { return rope_rows_body(a, s); }
extern "C" int r4d_rad_quant_roundtrip(const RadArgs* a, RadStream s) { return quant_rt_body(a, s); }
extern "C" int r4d_rad_mhc_mix(const RadArgs* a, RadStream s) { return mhc_mix_body(a, s); }
/* Partials of every chunk of every row of the band, so the projection spreads over workgroups. */
extern "C" int64_t r4d_rad_mhc_mix_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 2 || hc > 8 || hc * n > (1LL << 30)) return 0;
    return (int64_t)(M * mix_chunks((long)(hc * n)) * (hc * (hc + 2) + 1) * (long long)sizeof(float));
}
extern "C" int r4d_rad_mhc_collapse(const RadArgs* a, RadStream s) {
    return mhc_collapse_body(a, s);
}
extern "C" int r4d_rad_softmax_pool(const RadArgs* a, RadStream s) {
    return softmax_pool_body(a, s);
}
extern "C" int r4d_rad_entry_store(const RadArgs* a, RadStream s) { return entry_store_body(a, s); }
extern "C" int r4d_rad_router_scored_topk(const RadArgs* a, RadStream s) {
    return router_scored_body(a, s);
}
extern "C" int r4d_rad_router_scored_topk_vl(const RadArgs* a, RadStream s) {
    return router_scored_vl_body(a, s);
}
extern "C" int r4d_rad_ds41_vit_rope2d(const RadArgs* a, RadStream s) {
    return vit_rope2d_body(a, s);
}
extern "C" int r4d_rad_ds41_pack3x3(const RadArgs* a, RadStream s) { return pack3x3_body(a, s); }
extern "C" int r4d_rad_ds41_span_rows(const RadArgs* a, RadStream s) { return span_rows_body(a, s); }
extern "C" int r4d_rad_embed_lookup_planes(const RadArgs* a, RadStream s) {
    return embed_planes_body(a, s);
}
extern "C" int r4d_rad_rmsnorm_rope_roundtrip(const RadArgs* a, RadStream s) {
    return norm_rope_rt_body(a, s);
}
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs* a, RadStream s) {
    return rope_rt_body(a, s);
}
extern "C" int r4d_rad_mhc_collapse_rmsnorm(const RadArgs* a, RadStream s) {
    return collapse_norm_body(a, s);
}
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs* a, RadStream s) {
    return post_collapse_norm_body(a, s);
}
