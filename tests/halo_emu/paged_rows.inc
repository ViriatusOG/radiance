extern "C" {
int halo_rad_attn_decode_h128_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa1_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa1_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa2_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa2_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa2w_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa2w_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa8_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h128_gqa8_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa2_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa2_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa4_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa4_fp8(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa8_bf16(const RadArgs*, RadStream);
int halo_rad_attn_decode_h256_gqa8_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa1_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa1_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa2_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa2_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa2w_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa2w_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa8_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h128_gqa8_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa2_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa2_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa4_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa4_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa8_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h256_gqa8_fp8(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h64_gqa4_bf16(const RadArgs*, RadStream);
int halo_rad_attn_prefill_h64_gqa4_fp8(const RadArgs*, RadStream);
int64_t halo_rad_attn_prefill_h128_gqa1_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h128_gqa2_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h128_gqa8_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h256_gqa4_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h256_gqa8_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h256_scratch(const RadArgs*);
int64_t halo_rad_attn_prefill_h64_gqa4_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h128_gqa1_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h128_gqa2_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h128_gqa2w_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h128_gqa8_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h128_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h256_gqa2_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h256_gqa4_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h256_gqa8_scratch(const RadArgs*);
int64_t halo_rad_attn_decode_h256_scratch(const RadArgs*);
}
static const Row kRows[] = {
    { "attn_prefill_h256_gqa6_fp8kv", halo_rad_attn_prefill_h256_fp8, halo_rad_attn_prefill_h256_scratch, 256, 6, true, 1048576, 0, 1 },
    { "attn_prefill_h256_gqa6_bf16kv", halo_rad_attn_prefill_h256_bf16, halo_rad_attn_prefill_h256_scratch, 256, 6, false, 1048576, 0, 1 },
    { "attn_decode_h256_gqa6_fp8kv", halo_rad_attn_decode_h256_fp8, halo_rad_attn_decode_h256_scratch, 256, 6, true, 10, 0, 1 },
    { "attn_decode_h256_gqa6_bf16kv", halo_rad_attn_decode_h256_bf16, halo_rad_attn_decode_h256_scratch, 256, 6, false, 10, 0, 1 },
    { "attn_decode_h128_gqa4_fp8kv", halo_rad_attn_decode_h128_fp8, halo_rad_attn_decode_h128_scratch, 128, 4, true, 16, 1, 0 },
    { "attn_decode_h128_gqa4_bf16kv", halo_rad_attn_decode_h128_bf16, halo_rad_attn_decode_h128_scratch, 128, 4, false, 16, 1, 0 },
    { "attn_prefill_h256_gqa2_fp8kv", halo_rad_attn_prefill_h256_gqa2_fp8, nullptr, 256, 2, true, 1048576, 1, 1 },
    { "attn_prefill_h256_gqa2_bf16kv", halo_rad_attn_prefill_h256_gqa2_bf16, nullptr, 256, 2, false, 1048576, 1, 1 },
    { "attn_decode_h256_gqa2_fp8kv", halo_rad_attn_decode_h256_gqa2_fp8, halo_rad_attn_decode_h256_gqa2_scratch, 256, 2, true, 32, 1, 0 },
    { "attn_decode_h256_gqa2_bf16kv", halo_rad_attn_decode_h256_gqa2_bf16, halo_rad_attn_decode_h256_gqa2_scratch, 256, 2, false, 32, 1, 0 },
    { "attn_prefill_h128_gqa2w_fp8kv", halo_rad_attn_prefill_h128_gqa2w_fp8, nullptr, 128, 2, true, 1048576, 2, 1 },
    { "attn_prefill_h128_gqa2w_bf16kv", halo_rad_attn_prefill_h128_gqa2w_bf16, nullptr, 128, 2, false, 1048576, 2, 1 },
    { "attn_decode_h128_gqa2w_fp8kv", halo_rad_attn_decode_h128_gqa2w_fp8, halo_rad_attn_decode_h128_gqa2w_scratch, 128, 2, true, 32, 2, 0 },
    { "attn_decode_h128_gqa2w_bf16kv", halo_rad_attn_decode_h128_gqa2w_bf16, halo_rad_attn_decode_h128_gqa2w_scratch, 128, 2, false, 32, 2, 0 },
    { "attn_prefill_h128_gqa1_fp8kv", halo_rad_attn_prefill_h128_gqa1_fp8, halo_rad_attn_prefill_h128_gqa1_scratch, 128, 1, true, 1048576, 0, 1 },
    { "attn_prefill_h128_gqa1_bf16kv", halo_rad_attn_prefill_h128_gqa1_bf16, halo_rad_attn_prefill_h128_gqa1_scratch, 128, 1, false, 1048576, 0, 1 },
    { "attn_decode_h128_gqa1_fp8kv", halo_rad_attn_decode_h128_gqa1_fp8, halo_rad_attn_decode_h128_gqa1_scratch, 128, 1, true, 64, 0, 0 },
    { "attn_decode_h128_gqa1_bf16kv", halo_rad_attn_decode_h128_gqa1_bf16, halo_rad_attn_decode_h128_gqa1_scratch, 128, 1, false, 64, 0, 0 },
    { "attn_prefill_h256_gqa8_fp8kv", halo_rad_attn_prefill_h256_gqa8_fp8, halo_rad_attn_prefill_h256_gqa8_scratch, 256, 8, true, 1048576, 0, 1 },
    { "attn_prefill_h256_gqa8_bf16kv", halo_rad_attn_prefill_h256_gqa8_bf16, halo_rad_attn_prefill_h256_gqa8_scratch, 256, 8, false, 1048576, 0, 1 },
    { "attn_decode_h256_gqa8_fp8kv", halo_rad_attn_decode_h256_gqa8_fp8, halo_rad_attn_decode_h256_gqa8_scratch, 256, 8, true, 8, 0, 0 },
    { "attn_decode_h256_gqa8_bf16kv", halo_rad_attn_decode_h256_gqa8_bf16, halo_rad_attn_decode_h256_gqa8_scratch, 256, 8, false, 8, 0, 0 },
    { "attn_prefill_h256_gqa4_fp8kv", halo_rad_attn_prefill_h256_gqa4_fp8, halo_rad_attn_prefill_h256_gqa4_scratch, 256, 4, true, 1048576, 0, 1 },
    { "attn_prefill_h256_gqa4_bf16kv", halo_rad_attn_prefill_h256_gqa4_bf16, halo_rad_attn_prefill_h256_gqa4_scratch, 256, 4, false, 1048576, 0, 1 },
    { "attn_decode_h256_gqa4_fp8kv", halo_rad_attn_decode_h256_gqa4_fp8, halo_rad_attn_decode_h256_gqa4_scratch, 256, 4, true, 16, 0, 0 },
    { "attn_decode_h256_gqa4_bf16kv", halo_rad_attn_decode_h256_gqa4_bf16, halo_rad_attn_decode_h256_gqa4_scratch, 256, 4, false, 16, 0, 0 },
    { "attn_prefill_h64_gqa4_fp8kv", halo_rad_attn_prefill_h64_gqa4_fp8, halo_rad_attn_prefill_h64_gqa4_scratch, 64, 4, true, 1048576, 0, 1 },
    { "attn_prefill_h64_gqa4_bf16kv", halo_rad_attn_prefill_h64_gqa4_bf16, halo_rad_attn_prefill_h64_gqa4_scratch, 64, 4, false, 1048576, 0, 1 },
    { "attn_prefill_h128_gqa8_fp8kv", halo_rad_attn_prefill_h128_gqa8_fp8, halo_rad_attn_prefill_h128_gqa8_scratch, 128, 8, true, 1048576, 0, 1 },
    { "attn_prefill_h128_gqa8_bf16kv", halo_rad_attn_prefill_h128_gqa8_bf16, halo_rad_attn_prefill_h128_gqa8_scratch, 128, 8, false, 1048576, 0, 1 },
    { "attn_decode_h128_gqa8_fp8kv", halo_rad_attn_decode_h128_gqa8_fp8, halo_rad_attn_decode_h128_gqa8_scratch, 128, 8, true, 8, 0, 0 },
    { "attn_decode_h128_gqa8_bf16kv", halo_rad_attn_decode_h128_gqa8_bf16, halo_rad_attn_decode_h128_gqa8_scratch, 128, 8, false, 8, 0, 0 },
    { "attn_prefill_h128_gqa2_fp8kv", halo_rad_attn_prefill_h128_gqa2_fp8, halo_rad_attn_prefill_h128_gqa2_scratch, 128, 2, true, 1048576, 0, 1 },
    { "attn_prefill_h128_gqa2_bf16kv", halo_rad_attn_prefill_h128_gqa2_bf16, halo_rad_attn_prefill_h128_gqa2_scratch, 128, 2, false, 1048576, 0, 1 },
    { "attn_decode_h128_gqa2_fp8kv", halo_rad_attn_decode_h128_gqa2_fp8, halo_rad_attn_decode_h128_gqa2_scratch, 128, 2, true, 32, 0, 0 },
    { "attn_decode_h128_gqa2_bf16kv", halo_rad_attn_decode_h128_gqa2_bf16, halo_rad_attn_decode_h128_gqa2_scratch, 128, 2, false, 32, 0, 0 },
};
