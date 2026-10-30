/* tests/halo_emu: libhalo's elementwise and data-movement rows on the host SIMT emulator, each held
 * to libref's row for the same op on the same operands.
 *
 * These are the image, video, audio and residual-stream helpers no model fixture of rad-kbench
 * reaches and halo_selftest has no case for (gelu_tanh, cast, the row scatters and gathers,
 * cu_repeat, randn_lanes, to_rgba8, the depth/space shuffles, ln_mod, l2norm_chan, chan_split,
 * resample_nearest, fourier_embed, end_status, hc_enter/hc_write). Operands come from libref's own
 * operand descriptions (RadShapeFn) at a hand-picked geometry -- the schema is shared, so its
 * descriptors are libhalo's too -- drawn as rad-kbench draws them; every written operand is
 * compared byte for byte, except where a case names a bound: there the two sides reach the value
 * by a different summation order (a row norm over 32 lanes against libref's loop) and one bf16
 * rounding may flip, so floats compare by rel_l2. On the emulator both sides use the host's libm,
 * so a transcendental is the same function on both.
 *
 * libref is opened from $RADIANCE_HOME/kernels; without it the test fails (it has nothing else to
 * hold the rows to). */
#include "hip/hip_runtime.h"   // tests/r4d_emu/emu: the emulator
#include "rad_abi.h"
#include "rad_plugin.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
int halo_rad_gelu_tanh(const RadArgs*, RadStream);
int halo_rad_cast_bf16(const RadArgs*, RadStream);
int halo_rad_cast_widen(const RadArgs*, RadStream);
int halo_rad_scatter_rows_bf16(const RadArgs*, RadStream);
int halo_rad_cu_repeat(const RadArgs*, RadStream);
int halo_rad_randn_lanes(const RadArgs*, RadStream);
int halo_rad_to_rgba8(const RadArgs*, RadStream);
int halo_rad_d2s_add(const RadArgs*, RadStream);
int halo_rad_s2d_avg_add(const RadArgs*, RadStream);
int halo_rad_ln_mod(const RadArgs*, RadStream);
int halo_rad_l2norm_chan(const RadArgs*, RadStream);
int halo_rad_gather_rows2(const RadArgs*, RadStream);
int halo_rad_scatter_add_rows(const RadArgs*, RadStream);
int halo_rad_chan_split(const RadArgs*, RadStream);
int halo_rad_resample_nearest(const RadArgs*, RadStream);
int halo_rad_fourier_embed(const RadArgs*, RadStream);
int halo_rad_end_status(const RadArgs*, RadStream);
int halo_rad_hc_enter(const RadArgs*, RadStream);
int halo_rad_hc_write(const RadArgs*, RadStream);
}

namespace {

struct Ref {
    void* h = nullptr;
    int (*kc)(void) = nullptr;
    const RadKernelInfo* (*ka)(int) = nullptr;
    int (*sc)(void) = nullptr;
    const RadOpSchema* (*sa)(int) = nullptr;
    const RadKernelInfo* row(const char* op) const {
        for (int i = 0; i < kc(); ++i)
            if (!std::strcmp(ka(i)->op, op)) return ka(i);
        return nullptr;
    }
    const RadOpSchema* schema(const char* op) const {
        for (int i = 0; i < sc(); ++i)
            if (!std::strcmp(sa(i)->op, op)) return sa(i);
        return nullptr;
    }
};

struct Param {
    const char* key;
    long long i;
    const char* s;
    double f;
    int kind;
};
Param I(const char* k, long long v) { return { k, v, nullptr, 0.0, RAD_P_INT }; }
Param S(const char* k, const char* v) { return { k, 0, v, 0.0, RAD_P_STR }; }
Param F(const char* k, double v) { return { k, 0, nullptr, v, RAD_P_F64 }; }

struct Case {
    const char* name;
    const char* op;
    int (*halo)(const RadArgs*, RadStream);
    std::vector<Param> p;
    /* Operands libref declines to describe (a signed index): given here, then adjusted. */
    std::vector<RadOpdDesc> opds = {};
    void (*fix)(std::vector<std::vector<char>>&) = nullptr;
    double tol = 0.0;   /* 0: every written byte equal; > 0: float outputs within this rel_l2 */
};

RadOpdDesc desc(uint32_t dt, std::vector<int64_t> shape, uint32_t fill = RAD_FILL_NORMAL,
                int64_t idx_max = 0) {
    RadOpdDesc d;
    std::memset(&d, 0, sizeof d);
    d.dtype = dt; d.rank = (uint32_t)shape.size(); d.fill = fill; d.idx_max = idx_max; d.idx_const = -1;
    for (size_t i = 0; i < shape.size(); ++i) d.shape[i] = shape[i];
    return d;
}

std::mt19937_64 rng(97);

size_t bytes_of(const RadOpdDesc& d) {
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    return (size_t)((n * rad_dtype_bits(d.dtype) + 7) / 8);
}
int64_t numel(const RadOpdDesc& d) {
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    return n;
}

/* Draws one operand as rad-kbench does for the fill kinds these ops use. */
bool draw(const RadOpdDesc& d, std::vector<char>& buf, int64_t prev_lead) {
    const int64_t n = numel(d);
    buf.assign(bytes_of(d) + 64, 0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.02f, 0.98f);
    auto put_f = [&](int64_t i, float v) {
        switch (d.dtype) {
          case RAD_F32: std::memcpy(buf.data() + 4 * i, &v, 4); break;
          case RAD_BF16: { const uint16_t b = rad_f32_to_bf16(v); std::memcpy(buf.data() + 2 * i, &b, 2); break; }
          case RAD_F16: { const _Float16 h = (_Float16)v; std::memcpy(buf.data() + 2 * i, &h, 2); break; }
          default: break;
        }
    };
    auto put_i = [&](int64_t i, int64_t v) {
        switch (d.dtype) {
          case RAD_I32: case RAD_U32: { const int32_t x = (int32_t)v; std::memcpy(buf.data() + 4 * i, &x, 4); break; }
          case RAD_I64: std::memcpy(buf.data() + 8 * i, &v, 8); break;
          case RAD_U8: case RAD_I8: case RAD_BOOL: buf[(size_t)i] = (char)v; break;
          case RAD_I16: { const int16_t x = (int16_t)v; std::memcpy(buf.data() + 2 * i, &x, 2); break; }
          default: break;
        }
    };
    const bool is_float = d.dtype == RAD_F32 || d.dtype == RAD_BF16 || d.dtype == RAD_F16;
    switch (d.fill) {
      case RAD_FILL_NORMAL:
        if (is_float) for (int64_t i = 0; i < n; ++i) put_f(i, nd(rng));
        else for (int64_t i = 0; i < n; ++i) put_i(i, (int64_t)(rng() % 7));
        return true;
      case RAD_FILL_SIGMOID:
        for (int64_t i = 0; i < n; ++i) put_f(i, ud(rng));
        return true;
      case RAD_FILL_INDEX: {
        const int64_t hi = d.idx_max > 0 ? d.idx_max : prev_lead;
        if (d.idx_const >= 0) { for (int64_t i = 0; i < n; ++i) put_i(i, d.idx_const); return true; }
        if (d.flags & RAD_OPD_F_IDX_CU) {
            /* [0, ..., hi]: ragged segment ends, strictly non-decreasing */
            std::vector<int64_t> cut((size_t)n);
            cut[0] = 0;
            for (int64_t i = 1; i < n; ++i) cut[(size_t)i] = (int64_t)(rng() % (uint64_t)(hi + 1));
            std::sort(cut.begin() + 1, cut.end());
            cut[(size_t)(n - 1)] = hi;
            for (int64_t i = 0; i < n; ++i) put_i(i, cut[(size_t)i]);
            return true;
        }
        if (d.flags & RAD_OPD_F_IDX_UNIQUE) {
            if (hi < n) return false;
            std::vector<int64_t> perm((size_t)hi);
            for (int64_t i = 0; i < hi; ++i) perm[(size_t)i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int64_t i = 0; i < n; ++i) put_i(i, perm[(size_t)i]);
            return true;
        }
        if (hi <= 0) return false;
        for (int64_t i = 0; i < n; ++i) put_i(i, (int64_t)(rng() % (uint64_t)hi));
        return true;
      }
      default:
        std::printf("    fill kind %u not drawn here\n", d.fill);
        return false;
    }
}

double as_double(const RadOpdDesc& d, const std::vector<char>& b, int64_t i) {
    switch (d.dtype) {
      case RAD_F32: { float v; std::memcpy(&v, b.data() + 4 * i, 4); return v; }
      case RAD_BF16: { uint16_t v; std::memcpy(&v, b.data() + 2 * i, 2); return rad_bf16_to_f32(v); }
      case RAD_F16: { _Float16 v; std::memcpy(&v, b.data() + 2 * i, 2); return (double)(float)v; }
      default: return 0.0;
    }
}

int run_case(const Ref& ref, const Case& c) {
    const RadKernelInfo* rr = ref.row(c.op);
    const RadOpSchema* sch = ref.schema(c.op);
    if (!rr || !sch || !rr->opd_shape) {
        std::printf("FAIL %-34s libref has no row/schema/shape hook for %s\n", c.name, c.op);
        return 1;
    }
    std::vector<RadParam> ps;
    for (const Param& q : c.p) {
        RadParam r; std::memset(&r, 0, sizeof r);
        r.key = q.key; r.kind = q.kind; r.ival = q.i; r.sval = q.s; r.dval = q.f;
        ps.push_back(r);
    }
    const int no = sch->n_operands;
    std::vector<RadOpdDesc> d((size_t)no);
    std::vector<std::vector<char>> in((size_t)no);
    int64_t prev_lead = 0;
    for (int i = 0; i < no; ++i) {
        std::memset(&d[(size_t)i], 0, sizeof(RadOpdDesc));
        d[(size_t)i].idx_const = -1;
        if (!c.opds.empty()) d[(size_t)i] = c.opds[(size_t)i];
        else if (rr->opd_shape(ps.data(), (int)ps.size(), i, &d[(size_t)i]) != RAD_OK) {
            std::printf("FAIL %-34s libref's shape hook refused operand %d\n", c.name, i);
            return 1;
        }
        if (d[(size_t)i].flags & RAD_OPD_F_ABSENT) continue;
        if (!draw(d[(size_t)i], in[(size_t)i], prev_lead)) {
            std::printf("FAIL %-34s cannot draw operand %d (%s)\n", c.name, i, sch->operands[i].name);
            return 1;
        }
        if (d[(size_t)i].rank) prev_lead = d[(size_t)i].shape[0];
    }
    if (c.fix) c.fix(in);
    /* Two copies of every operand; outputs start from the same bytes (an INOUT is read too). */
    std::vector<std::vector<char>> a = in, b = in;
    auto tensors = [&](std::vector<std::vector<char>>& bufs) {
        std::vector<RadTensor> ts((size_t)no);
        for (int i = 0; i < no; ++i) {
            RadTensor& t = ts[(size_t)i];
            std::memset(&t, 0, sizeof t);
            const RadOpdDesc& e = d[(size_t)i];
            if (e.flags & RAD_OPD_F_ABSENT) continue;
            t.data = bufs[(size_t)i].data();
            t.dtype = e.dtype;
            t.rank = e.rank;
            int64_t acc = 1;
            for (int k = (int)e.rank - 1; k >= 0; --k) { t.shape[k] = e.shape[k]; t.stride[k] = acc; acc *= e.shape[k]; }
        }
        return ts;
    };
    std::vector<RadTensor> ta = tensors(a), tb = tensors(b);
    RadArgs ra; std::memset(&ra, 0, sizeof ra);
    ra.t = ta.data(); ra.n_t = no; ra.p = ps.data(); ra.n_p = (int)ps.size(); ra.world_size = 1;
    RadArgs rb = ra;
    rb.t = tb.data();
    std::vector<char> sa, sb;
    if (rr->scratch) {
        const int64_t need = rr->scratch(&rb);
        if (need > 0) { sb.assign((size_t)need, 0); rb.scratch = sb.data(); rb.scratch_bytes = need; }
    }
    const int rh = c.halo(&ra, nullptr);
    const int rf = rr->launch(&rb, nullptr);
    if (rh != RAD_OK || rf != RAD_OK) {
        std::printf("FAIL %-34s rc libhalo %d, libref %d\n", c.name, rh, rf);
        return 1;
    }
    int fails = 0;
    std::string detail;
    for (int i = 0; i < no; ++i) {
        const int role = sch->operands[i].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        const RadOpdDesc& e = d[(size_t)i];
        if (e.flags & RAD_OPD_F_ABSENT) continue;
        const int64_t n = numel(e);
        char buf[160];
        if (c.tol > 0 && (e.dtype == RAD_F32 || e.dtype == RAD_BF16 || e.dtype == RAD_F16)) {
            double num = 0, den = 0, worst = 0;
            int64_t nonfinite = 0;
            for (int64_t k = 0; k < n; ++k) {
                const double x = as_double(e, a[(size_t)i], k), y = as_double(e, b[(size_t)i], k);
                if (std::isfinite(x) != std::isfinite(y)) { ++nonfinite; continue; }
                if (!std::isfinite(y)) continue;
                num += (x - y) * (x - y); den += y * y;
                worst = std::max(worst, std::fabs(x - y));
            }
            const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
            /* An all-zero reference would pass any kernel that writes nothing: not a check. */
            const bool ok = nonfinite == 0 && rel < c.tol && den > 0;
            std::snprintf(buf, sizeof buf, " %s: rel_l2 %.2e max_abs %.2e%s%s", sch->operands[i].name, rel, worst,
                          nonfinite ? " NONFINITE MISMATCH" : "", den > 0 ? "" : " REFERENCE ALL ZERO");
            detail += buf;
            fails += !ok;
        } else {
            const size_t nb = bytes_of(e);
            size_t diff = 0, nonzero = 0;
            for (size_t k = 0; k < nb; ++k) { diff += a[(size_t)i][k] != b[(size_t)i][k]; nonzero += b[(size_t)i][k] != 0; }
            std::snprintf(buf, sizeof buf, " %s: %zu of %zu bytes differ%s", sch->operands[i].name, diff, nb,
                          nonzero ? "" : " REFERENCE ALL ZERO");
            detail += buf;
            fails += diff != 0 || nonzero == 0;
        }
    }
    std::printf("%s %-34s%s\n", fails ? "FAIL" : "ok  ", c.name, detail.c_str());
    return fails ? 1 : 0;
}

}  // namespace

int main() {
    const char* home = std::getenv("RADIANCE_HOME");
    const std::string path = std::string(home ? home : ".") + "/kernels/libref.so";
    Ref ref;
    ref.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ref.h) { std::printf("FAIL cannot open %s: %s\n", path.c_str(), dlerror()); return 1; }
    ref.kc = (int (*)(void))dlsym(ref.h, "rad_kernel_count");
    ref.ka = (const RadKernelInfo* (*)(int))dlsym(ref.h, "rad_kernel_at");
    ref.sc = (int (*)(void))dlsym(ref.h, "rad_kernel_schema_count");
    ref.sa = (const RadOpSchema* (*)(int))dlsym(ref.h, "rad_kernel_schema_at");
    if (!ref.kc || !ref.ka || !ref.sc || !ref.sa) { std::printf("FAIL libref has no tables\n"); return 1; }

    const std::vector<Case> cases = {
        { "gelu_tanh bf16 M 7 n 1032", "gelu_tanh", halo_rad_gelu_tanh, { I("M", 7), I("n", 1032), S("dtype", "bf16") } },
        { "cast f32->bf16 M 5 n 777", "cast", halo_rad_cast_bf16, { I("M", 5), I("n", 777), S("from", "f32"), S("to", "bf16") } },
        { "cast bf16->f32 M 5 n 777", "cast", halo_rad_cast_widen, { I("M", 5), I("n", 777), S("from", "bf16"), S("to", "f32") } },
        { "scatter_rows bf16 M 9 n 258", "scatter_rows", halo_rad_scatter_rows_bf16, { I("M", 9), I("n", 258), S("dtype", "bf16") } },
        { "cu_repeat M 6 mult 3", "cu_repeat", halo_rad_cu_repeat, { I("M", 6), I("mult", 3) } },
        { "randn_lanes bf16 M 12 n 64", "randn_lanes", halo_rad_randn_lanes, { I("M", 12), I("n", 64), I("stride", 64), S("dtype", "bf16") } },
        { "randn_lanes f32 M 12 n 96 stride 100", "randn_lanes", halo_rad_randn_lanes, { I("M", 12), I("n", 96), I("stride", 100), S("dtype", "f32") } },
        { "to_rgba8 bf16 M 33 n 3", "to_rgba8", halo_rad_to_rgba8, { I("M", 33), I("n", 3), S("dtype", "bf16") } },
        { "d2s_add bf16 M 4 cin 64 cout 32 ft 2", "d2s_add", halo_rad_d2s_add, { I("M", 4), I("cin", 64), I("cout", 32), I("ft", 2), S("dtype", "bf16") } },
        { "s2d_avg_add bf16 M 4 cin 32 cout 64 ft 2", "s2d_avg_add", halo_rad_s2d_avg_add, { I("M", 4), I("cin", 32), I("cout", 64), I("ft", 2), S("dtype", "bf16") } },
        { "ln_mod bf16 M 10 n 1032", "ln_mod", halo_rad_ln_mod, { I("M", 10), I("n", 1032), F("eps", 1e-6), S("dtype", "bf16") } },
        { "l2norm_chan bf16 M 9 n 520", "l2norm_chan", halo_rad_l2norm_chan, { I("M", 9), I("n", 520), S("act", "none"), S("dtype", "bf16") } },
        { "l2norm_chan bf16 silu M 9 n 520", "l2norm_chan", halo_rad_l2norm_chan, { I("M", 9), I("n", 520), S("act", "silu"), S("dtype", "bf16") }, {}, nullptr, 1e-2 },
        /* y[i] = idx >= 0 ? a[idx] : b[-1 - idx]: a 7-row and a 5-row source, the index drawn over
         * [0, 12) and moved to [-5, 7) so both sources are read. */
        { "gather_rows2 bf16 M 11 n 264", "gather_rows2", halo_rad_gather_rows2, { I("M", 11), I("n", 264), S("dtype", "bf16") },
          { desc(RAD_BF16, { 7, 264 }), desc(RAD_BF16, { 5, 264 }), desc(RAD_I32, { 11 }, RAD_FILL_INDEX, 12),
            desc(RAD_BF16, { 11, 264 }) },
          [](std::vector<std::vector<char>>& b) {
              for (int i = 0; i < 11; ++i) { int32_t v; std::memcpy(&v, b[2].data() + 4 * i, 4); v -= 5; std::memcpy(b[2].data() + 4 * i, &v, 4); }
          } },
        { "scatter_add_rows bf16 M 9 n 264", "scatter_add_rows", halo_rad_scatter_add_rows, { I("M", 9), I("n", 264), S("dtype", "bf16") } },
        { "chan_split bf16 M 6 n 96 parts 3", "chan_split", halo_rad_chan_split, { I("M", 6), I("n", 96), I("parts", 3), S("dtype", "bf16") } },
        { "resample_nearest bf16 M 7 n 48", "resample_nearest", halo_rad_resample_nearest, { I("M", 7), I("n", 48), S("dtype", "bf16") } },
        { "fourier_embed bf16 M 5 n 64", "fourier_embed", halo_rad_fourier_embed, { I("M", 5), I("n", 64), I("round", 0), S("dtype", "bf16") } },
        { "end_status M 4 end 3 rep 2", "end_status", halo_rad_end_status, { I("M", 4), I("end", 3), I("rep", 2) } },
        { "end_status M 3 end 5 rep 1", "end_status", halo_rad_end_status, { I("M", 3), I("end", 5), I("rep", 1) } },
        { "hc_enter bf16 M 5 n 520 hc 4", "hc_enter", halo_rad_hc_enter, { I("M", 5), I("n", 520), I("hc", 4), S("dtype", "bf16") } },
        { "hc_write bf16 M 5 n 520 hc 4", "hc_write", halo_rad_hc_write, { I("M", 5), I("n", 520), I("hc", 4), S("dtype", "bf16") }, {}, nullptr, 1e-2 },
    };
    int fails = 0;
    for (const Case& c : cases) fails += run_case(ref, c);
    std::printf("libhalo vs libref: %zu case(s), %d failure(s)\n", cases.size(), fails);
    return fails ? 1 : 0;
}
