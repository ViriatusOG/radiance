// Inkling 2's recurrent DSpark Markov walk (`dspark_rnn_markov_greedy`, markov_head_type
// rnn_hidden). dflash_markov_greedy's bigram bias is a fixed function of the previous token; here a
// rank-R state carries the walk, so each position runs three kernels in turn:
//
//   rnn_ctx     once: every position's [hidden, frnn] context through rnn_w's last 2E columns
//   rnn_step    the state update from [state, w1[prev]] and the latent tanh(out)
//   score/reduce the full-vocabulary scan with the folded head (lm_head @ hidden_out): the score is
//               softcap(bf16(bf16(logit) + bf16(fold_b + latent . fold_w))), then the exact argmax
//
// Every step rounds to bf16 where the reference's torch ops do (libref/ref_quant.cpp spells the
// arithmetic out). The scan's tile decomposition is dflash_markov_greedy's: 64 ids a CTA, the
// local winners reduced before the next position starts.
#include <hip/hip_runtime.h>
#include <climits>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"  // r4d_bf2f, r4d_f2bf, r4d_u16
#include "r4d_common.h"           // r4d_wave_sum

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWarps = kThreads / kWave;
constexpr int kCandidatesPerTile = 64;
constexpr int kCandidatesPerWarp = kCandidatesPerTile / kWarps;
constexpr int kMaxRank = 512;
constexpr int kMaxSteps = 16;

__device__ __forceinline__ float bfr(float x) { return r4d_bf2f(r4d_f2bf(x)); }

// ctx[m][l][j] = rnn_b[j] + rnn_w[j, 2R:2R+E] . hidden[m*steps+l] + rnn_w[j, 2R+E:] . frnn[m].
// One warp an output j, over every position of the sequence, so rnn_w's row is read once.
__global__ __launch_bounds__(kThreads) void rnn_ctx(
    const r4d_u16* __restrict__ hidden, long hidden_pitch,
    const r4d_u16* __restrict__ frnn, long frnn_pitch,
    const r4d_u16* __restrict__ rnn_w, const r4d_u16* __restrict__ rnn_b,
    float* __restrict__ scratch, long scratch_pitch, int steps, int rank, int embd) {
  const int seq = blockIdx.y;
  const int lane = threadIdx.x & (kWave - 1);
  const int j = blockIdx.x * kWarps + threadIdx.x / kWave;
  const int R3 = 3 * rank;
  if (j >= R3) return;
  const long W = 2L * rank + 2L * embd;
  const r4d_u16* wrow = rnn_w + (long)j * W + 2L * rank;
  float acc[kMaxSteps];
#pragma unroll
  for (int l = 0; l < kMaxSteps; ++l) acc[l] = 0.0f;
  float fsum = 0.0f;
  for (int i = lane; i < embd; i += kWave) {
    const float wh = r4d_bf2f(wrow[i]);
    const float wf = r4d_bf2f(wrow[embd + i]);
    fsum += wf * r4d_bf2f(frnn[(long)seq * frnn_pitch + i]);
#pragma unroll
    for (int l = 0; l < kMaxSteps; ++l)
      if (l < steps) acc[l] += wh * r4d_bf2f(hidden[((long)seq * steps + l) * hidden_pitch + i]);
  }
  fsum = r4d_wave_sum(fsum);
  const float bias = r4d_bf2f(rnn_b[j]);
#pragma unroll
  for (int l = 0; l < kMaxSteps; ++l) {
    if (l >= steps) break;
    const float s = r4d_wave_sum(acc[l]);
    if (lane == 0) scratch[(long)seq * scratch_pitch + (long)l * R3 + j] = bias + s + fsum;
  }
}

// One CTA a sequence: raw = ctx[step] + rnn_w[:, :2R] @ [state, w1[prev]], then the gate. The
// state starts at zero at step 0; the predecessor is the anchor, then the walk's last pick.
__global__ __launch_bounds__(kThreads) void rnn_step(
    const int* __restrict__ anchor, long anchor_stride, const int* __restrict__ out, long out_pitch,
    const r4d_u16* __restrict__ w1, const r4d_u16* __restrict__ rnn_w,
    float* __restrict__ scratch, long scratch_pitch, int step, int steps, int rank, int embd,
    int vocab) {
  const int seq = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int warp = tid / kWave;
  const int R3 = 3 * rank;
  const long W = 2L * rank + 2L * embd;
  float* ctx = scratch + (long)seq * scratch_pitch;
  float* state = ctx + (long)steps * R3;
  float* latent = state + rank;
  const int prev = step == 0 ? anchor[(long)seq * anchor_stride]
                             : out[(long)seq * out_pitch + step - 1];
  const bool prev_ok = prev >= 0 && prev < vocab;

  __shared__ float s_in[2 * kMaxRank];
  __shared__ float s_raw[3 * kMaxRank];
  for (int r = tid; r < rank; r += kThreads) {
    s_in[r] = step == 0 ? 0.0f : state[r];
    s_in[rank + r] = prev_ok ? r4d_bf2f(w1[(long)prev * rank + r]) : 0.0f;
  }
  __syncthreads();
  for (int j = warp; j < R3; j += kWarps) {
    const r4d_u16* wrow = rnn_w + (long)j * W;
    float dot = 0.0f;
    for (int i = lane; i < 2 * rank; i += kWave) dot += r4d_bf2f(wrow[i]) * s_in[i];
    dot = r4d_wave_sum(dot);
    if (lane == 0) s_raw[j] = ctx[(long)step * R3 + j] + dot;
  }
  __syncthreads();
  for (int r = tid; r < rank; r += kThreads) {
    const float g = bfr(1.0f / (1.0f + expf(-bfr(s_raw[r]))));
    const float c = bfr(tanhf(bfr(s_raw[rank + r])));
    const float o = bfr(s_raw[2 * rank + r]);
    state[r] = bfr(bfr(g * s_in[r]) + bfr(bfr(1.0f - g) * c));
    latent[r] = bfr(tanhf(o));
  }
}

__global__ __launch_bounds__(kThreads) void rnn_score_tiles(
    const float* __restrict__ logits, long logits_pitch,
    const r4d_u16* __restrict__ fold_w, const r4d_u16* __restrict__ fold_b,
    const float* __restrict__ scratch, long scratch_pitch,
    float* __restrict__ tile_scores, long scores_pitch,
    int* __restrict__ tile_tokens, long tokens_pitch,
    int step, int steps, int rank, int vocab, float cap) {
  const int seq = blockIdx.y;
  const int tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int warp = tid / kWave;
  const float* latent = scratch + (long)seq * scratch_pitch + (long)steps * 3 * rank + rank;

  __shared__ float s_lat[kMaxRank];
  __shared__ float s_best[kWarps];
  __shared__ int s_token[kWarps];
  for (int r = tid; r < rank; r += kThreads) s_lat[r] = latent[r];
  __syncthreads();

  const float* lrow = logits + ((long)seq * steps + step) * logits_pitch;
  float warp_best = -INFINITY;
  int warp_token = tile * kCandidatesPerTile + warp;
#pragma unroll
  for (int jj = 0; jj < kCandidatesPerWarp; ++jj) {
    const int token = tile * kCandidatesPerTile + warp + jj * kWarps;
    if (token < vocab) {
      float dot = 0.0f;
#pragma unroll 8
      for (int r = lane; r < rank; r += kWave)
        dot += s_lat[r] * r4d_bf2f(fold_w[(long)token * rank + r]);
      dot = r4d_wave_sum(dot);
      if (lane == 0) {
        const float bias = bfr(r4d_bf2f(fold_b[token]) + dot);
        const float x = bfr(bfr(lrow[token]) + bias);
        float score = bfr(cap * tanhf(x / cap));
        if (score != score) score = -INFINITY;
        if (score > warp_best || (score == warp_best && token < warp_token)) {
          warp_best = score;
          warp_token = token;
        }
      }
    }
  }
  if (lane == 0) {
    s_best[warp] = warp_best;
    s_token[warp] = warp_token;
  }
  __syncthreads();
  if (tid == 0) {
    float best = -INFINITY;
    int best_token = tile * kCandidatesPerTile;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float score = s_best[w];
      const int token = s_token[w];
      if (score > best || (score == best && token < best_token)) {
        best = score;
        best_token = token;
      }
    }
    tile_scores[(long)seq * scores_pitch + tile] = best;
    tile_tokens[(long)seq * tokens_pitch + tile] = best_token;
  }
}

__global__ __launch_bounds__(kThreads) void rnn_reduce_tiles(
    const float* __restrict__ tile_scores, long scores_pitch,
    const int* __restrict__ tile_tokens, long tokens_pitch,
    int* __restrict__ out, long out_pitch, int step, int n_tiles) {
  const int seq = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_best[kThreads];
  __shared__ int s_token[kThreads];
  float best = -INFINITY;
  int best_token = INT_MAX;
  for (int tile = tid; tile < n_tiles; tile += kThreads) {
    const float score = tile_scores[(long)seq * scores_pitch + tile];
    const int token = tile_tokens[(long)seq * tokens_pitch + tile];
    if (score > best || (score == best && token < best_token)) {
      best = score;
      best_token = token;
    }
  }
  s_best[tid] = best;
  s_token[tid] = best_token;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float score = s_best[tid + stride];
      const int token = s_token[tid + stride];
      if (score > s_best[tid] || (score == s_best[tid] && token < s_token[tid])) {
        s_best[tid] = score;
        s_token[tid] = token;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[(long)seq * out_pitch + step] = s_token[0];
}

// The device computes element offsets in signed long; reject a row span that would overflow.
bool rows_fit(long long rows, long long pitch, long long width, int element_bytes) {
  const long long limit = LLONG_MAX / element_bytes;
  return rows > 0 && width > 0 && width <= limit && pitch >= width &&
         (rows == 1 || pitch <= (limit - (width - 1)) / (rows - 1));
}

}  // namespace

extern "C" int r4d_rad_dspark_rnn_markov_greedy(const RadArgs* a, RadStream stream) {
  enum { LOGITS, HIDDEN, FRNN, ANCHOR, W1, RNN_W, RNN_B, FOLD_W, FOLD_B, SCRATCH, TILE_SCORE,
         TILE_TOKEN, OUT };
  const RadTensor* t[13];
  for (int i = 0; i < 13; ++i) {
    t[i] = r4d_opt(a, i);
    if (!t[i]) return R4D_NO(RAD_E_INVAL);
  }
  const long long steps = r4d_geti_or(a, "steps", 0);
  const long long rank = r4d_geti_or(a, "rank", 0);
  const long long embd = r4d_geti_or(a, "n_embd", 0);
  const long long vocab = r4d_geti_or(a, "n_vocab", 0);
  const long long anchor_stride = r4d_geti_or(a, "anchor_stride", 1);
  const double cap = r4d_getf(a, "softcap", 0.0);
  if (steps <= 0 || steps > kMaxSteps || rank < kWave || rank > kMaxRank || rank % kWave ||
      embd <= 0 || embd > INT_MAX / 4 || vocab <= 0 || vocab > INT_MAX - kCandidatesPerTile ||
      anchor_stride <= 0 || !(cap > 0.0))
    return R4D_NO(RAD_E_INVAL);

  const RadTensor *logits = t[LOGITS], *hidden = t[HIDDEN], *frnn = t[FRNN], *anchor = t[ANCHOR];
  const RadTensor *out = t[OUT], *scratch = t[SCRATCH];
  if (out->dtype != RAD_I32 || out->rank != 2 || out->shape[1] < steps)
    return R4D_NO(RAD_E_SHAPE);
  const long long sequences = out->shape[0];
  const long long W = 2 * rank + 2 * embd, sw = steps * 3 * rank + 2 * rank;
  const long long n_tiles = (vocab + kCandidatesPerTile - 1) / kCandidatesPerTile;
  if (sequences <= 0 || sequences > 65535 ||
      logits->dtype != RAD_F32 || logits->rank != 2 || logits->shape[0] < sequences * steps ||
      logits->shape[1] < vocab ||
      !rows_fit(sequences * steps, r4d_row_pitch(logits), vocab, sizeof(float)) ||
      hidden->dtype != RAD_BF16 || hidden->rank != 2 || hidden->shape[0] < sequences * steps ||
      hidden->shape[1] < embd ||
      frnn->dtype != RAD_BF16 || frnn->rank != 2 || frnn->shape[0] < sequences ||
      frnn->shape[1] < embd ||
      anchor->dtype != RAD_I32 || !r4d_contig(anchor) ||
      r4d_numel(anchor) < (sequences - 1) * anchor_stride + 1 ||
      scratch->dtype != RAD_F32 || scratch->rank != 2 || scratch->shape[0] < sequences ||
      scratch->shape[1] < sw)
    return R4D_NO(RAD_E_SHAPE);
  for (int i : { W1, RNN_W, RNN_B, FOLD_W, FOLD_B })
    if (t[i]->dtype != RAD_BF16 || !r4d_contig(t[i])) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(t[W1]) < vocab * rank || r4d_numel(t[RNN_W]) < 3 * rank * W ||
      r4d_numel(t[RNN_B]) < 3 * rank || r4d_numel(t[FOLD_W]) < vocab * rank ||
      r4d_numel(t[FOLD_B]) < vocab)
    return R4D_NO(RAD_E_SHAPE);
  const RadTensor *scores = t[TILE_SCORE], *tokens = t[TILE_TOKEN];
  if (scores->dtype != RAD_F32 || tokens->dtype != RAD_I32 || scores->rank != 2 ||
      tokens->rank != 2 || scores->shape[0] < sequences || tokens->shape[0] < sequences ||
      scores->shape[1] < n_tiles || tokens->shape[1] < n_tiles)
    return R4D_NO(RAD_E_SHAPE);

  hipStream_t s = (hipStream_t)stream;
  const long sp = (long)r4d_row_pitch(scratch);
  const long op = (long)r4d_row_pitch(out);
  hipLaunchKernelGGL(rnn_ctx, dim3((unsigned)((3 * rank + kWarps - 1) / kWarps),
                                   (unsigned)sequences), dim3(kThreads), 0, s,
                     (const r4d_u16*)hidden->data, (long)r4d_row_pitch(hidden),
                     (const r4d_u16*)frnn->data, (long)r4d_row_pitch(frnn),
                     (const r4d_u16*)t[RNN_W]->data, (const r4d_u16*)t[RNN_B]->data,
                     (float*)scratch->data, sp, (int)steps, (int)rank, (int)embd);
  for (int step = 0; step < steps; ++step) {
    hipLaunchKernelGGL(rnn_step, dim3((unsigned)sequences), dim3(kThreads), 0, s,
                       (const int*)anchor->data, (long)anchor_stride, (const int*)out->data, op,
                       (const r4d_u16*)t[W1]->data, (const r4d_u16*)t[RNN_W]->data,
                       (float*)scratch->data, sp, step, (int)steps, (int)rank, (int)embd,
                       (int)vocab);
    hipLaunchKernelGGL(rnn_score_tiles, dim3((unsigned)n_tiles, (unsigned)sequences),
                       dim3(kThreads), 0, s, (const float*)logits->data,
                       (long)r4d_row_pitch(logits), (const r4d_u16*)t[FOLD_W]->data,
                       (const r4d_u16*)t[FOLD_B]->data, (const float*)scratch->data, sp,
                       (float*)scores->data, (long)r4d_row_pitch(scores), (int*)tokens->data,
                       (long)r4d_row_pitch(tokens), step, (int)steps, (int)rank, (int)vocab,
                       (float)cap);
    hipLaunchKernelGGL(rnn_reduce_tiles, dim3((unsigned)sequences), dim3(kThreads), 0, s,
                       (const float*)scores->data, (long)r4d_row_pitch(scores),
                       (const int*)tokens->data, (long)r4d_row_pitch(tokens), (int*)out->data,
                       op, step, (int)n_tiles);
  }
  return R4D_RAW((int)hipGetLastError());
}
