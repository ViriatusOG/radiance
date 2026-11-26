/* embed_exl3i_test.cpp -- the host check for embed_exl3i (embed_exl3i.h).
 *
 * Random exl3i tables (every codebook; integer, fractional and sub-1 rates; natural and R16 row order), every token
 * gathered by the host row and compared with the oracle: exl3i_decode_rows of the whole table, then one RNE bf16. They
 * must agree in every bit (the op's contract is the oracle's row). Then the layout hook against the encodings, and the
 * malformed calls the reader must refuse.
 */
#include "embed_exl3i.h"
#include "exl3i_encoding.h"
#include "exl3_decode.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int64_t bf16_key(uint16_t u) {
    const int64_t mag = u & 0x7FFF;
    return (u >> 15) ? -mag : mag;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

RadTensor tensor(void* data, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t acc = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = acc; acc *= t.shape[k]; }
    return t;
}

struct Table {
    int64_t nv, ne;
    int m, cb, r16;
    std::vector<uint32_t> rec;
    std::vector<uint16_t> suh, svh;
};

Table make(int64_t nv, int64_t ne, int m, int cb, int r16, unsigned seed) {
    std::mt19937 rng(seed);
    Table t{nv, ne, m, cb, r16, {}, {}, {}};
    t.rec.resize((size_t)(nv / 16 * exl3i_row_words(ne, m)));
    for (auto& w : t.rec) w = rng();
    std::uniform_real_distribution<double> mag(0.25, 2.0);
    auto sc = [&](int64_t n) {
        std::vector<uint16_t> v((size_t)n);
        for (auto& h : v) h = exl3_half_from_double((rng() & 1 ? -1.0 : 1.0) * mag(rng));
        return v;
    };
    t.suh = sc(ne);
    t.svh = sc(nv);
    return t;
}

struct Call {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
};

/* The params and operands of a gather of `tok` from table T into x (bf16 [M, ne]). */
void bind(Call& c, Table& T, std::vector<int32_t>& tok, std::vector<uint16_t>& x, const char* cbname) {
    const int64_t M = (int64_t)tok.size();
    c.p = { RAD_INT("M", M), RAD_INT("n_embd", T.ne), RAD_INT("n_vocab", T.nv), RAD_INT("m", T.m),
            RAD_STR("codebook", cbname), RAD_INT("r16", T.r16) };
    c.t = { tensor(tok.data(), RAD_I32, {M}),
            tensor(T.rec.data(), RAD_U32, {T.nv / 16, exl3i_row_words(T.ne, T.m)}),
            tensor(T.suh.data(), RAD_F16, {1, T.ne}),
            tensor(T.svh.data(), RAD_F16, {T.nv, 1}),
            tensor(x.data(), RAD_BF16, {M, T.ne}) };
    c.a.t = c.t.data();
    c.a.n_t = (int)c.t.size();
    c.a.p = c.p.data();
    c.a.n_p = (int)c.p.size();
}

void one(int64_t nv, int64_t ne, int m, int cb, int r16, unsigned seed) {
    Table T = make(nv, ne, m, cb, r16, seed);
    std::vector<int32_t> tok;
    for (int64_t v = 0; v < nv; ++v) tok.push_back((int32_t)v);
    tok.push_back(-1);
    tok.push_back((int32_t)nv);
    std::vector<uint16_t> x(tok.size() * (size_t)ne, 0xFFFF);
    Call c;
    bind(c, T, tok, x, exl3i_codebook_name(cb));
    const int rc = embed_exl3i_host_launch(&c.a, nullptr);
    char what[160];
    std::snprintf(what, sizeof what, "host launch m=%d cb=%s r16=%d: rc %d", m, exl3i_codebook_name(cb), r16, rc);
    check(rc == RAD_OK, what);
    if (rc != RAD_OK) return;

    Exl3iView v{};
    v.rows = nv; v.cols = ne; v.m = m; v.cb = cb;
    v.record = T.rec.data(); v.suh = T.suh.data(); v.svh = T.svh.data();
    std::vector<float> ref((size_t)(nv * ne));
    check(exl3i_decode_rows(&v, 0, nv, ref.data()) == 0, "oracle decode");
    int64_t off = 0, worse = 0;
    double rel = 0;   /* the largest |x - ref| / RMS(ref row) */
    for (int64_t id = 0; id < nv; ++id) {
        const int64_t n = embed_exl3i_row(id, nv, r16);
        double e2 = 0;
        for (int64_t k = 0; k < ne; ++k) e2 += (double)ref[(size_t)(n * ne + k)] * ref[(size_t)(n * ne + k)];
        const double rms = std::sqrt(e2 / (double)ne);
        for (int64_t k = 0; k < ne; ++k) {
            const uint16_t got = x[(size_t)(id * ne + k)], want = to_bf16(ref[(size_t)(n * ne + k)]);
            const int64_t d = bf16_key(got) - bf16_key(want);
            const uint32_t ug = (uint32_t)got << 16;
            float fg;
            std::memcpy(&fg, &ug, 4);
            const double r = std::fabs((double)fg - (double)ref[(size_t)(n * ne + k)]) / rms;
            off += d != 0;
            if (d > 1 || d < -1) {
                ++worse;
                if (r > rel) rel = r;
            }
        }
    }
    bool zeros = true;
    for (size_t i = (size_t)(nv * ne); i < x.size(); ++i) zeros = zeros && x[i] == 0;
    std::printf("m=%3d %-6s r16=%d: %lld of %lld elements off the oracle, %lld by > 1 ulp (at most %.2e of the row RMS)\n",
                m, exl3i_codebook_name(cb), r16, (long long)off, (long long)(nv * ne), (long long)worse, rel);
    std::snprintf(what, sizeof what, "m=%d cb=%s r16=%d: %lld off, %lld > 1 ulp up to %.2e RMS", m, exl3i_codebook_name(cb),
                  r16, (long long)off, (long long)worse, rel);
    check(off == 0, what);
    check(zeros, "an id outside [0, n_vocab) writes a zero row");
}

void layout_hook() {
    RadEncoding e{}, e16{};
    check(exl3i_encoding_make(64, EXL3I_CB_MAD, 0, 1024, 256, &e) == RAD_OK, "encoding");
    check(exl3i_encoding_make_r16(64, EXL3I_CB_MAD, 0, 1024, 256, &e16) == RAD_OK, "encoding r16");
    auto params = [](int m, const char* cb, int r16) {
        return std::vector<RadParam>{ RAD_INT("M", 4), RAD_INT("n_embd", 256), RAD_INT("n_vocab", 1024), RAD_INT("m", m),
                                      RAD_STR("codebook", cb), RAD_INT("r16", r16) };
    };
    const int sel_rec[1] = { EXL3I_PLANE_RECORD }, sel_suh[1] = { EXL3I_PLANE_SUH }, sel_svh[1] = { EXL3I_PLANE_SVH };
    auto p = params(64, "mad", 0);
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_RECORD, &e, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_UNSUPPORTED, "layout: the record as stored");
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_SUH, &e, sel_suh, nullptr, 0, nullptr) ==
              RAD_E_UNSUPPORTED, "layout: suh as stored");
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_SVH, &e, sel_svh, nullptr, 0, nullptr) ==
              RAD_E_UNSUPPORTED, "layout: svh as stored");
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_RECORD, &e, sel_suh, nullptr, 0, nullptr) ==
              RAD_E_SHAPE, "layout: the record operand on the suh plane");
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_RECORD, &e16, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_DTYPE, "layout: r16 encoding, natural params");
    auto p16 = params(64, "mad", 1);
    check(embed_exl3i_layout(p16.data(), (int)p16.size(), EMBED_EXL3I_RECORD, &e16, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_UNSUPPORTED, "layout: r16");
    auto pm = params(72, "mad", 0), pc = params(64, "other", 0);
    check(embed_exl3i_layout(pm.data(), (int)pm.size(), EMBED_EXL3I_RECORD, &e, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_DTYPE, "layout: another rate");
    check(embed_exl3i_layout(pc.data(), (int)pc.size(), EMBED_EXL3I_RECORD, &e, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_SHAPE, "layout: a codebook other than mad is refused (the parse names none)");
    {   /* a mad encoding lays out under codebook=mad (as stored) */
        RadEncoding em{};
        check(exl3i_encoding_make(64, EXL3I_CB_MAD, 0, 1024, 256, &em) == RAD_OK, "mad encoding");
        auto pmad = params(64, "mad", 0);
        check(embed_exl3i_layout(pmad.data(), (int)pmad.size(), EMBED_EXL3I_RECORD, &em, sel_rec, nullptr, 0, nullptr) ==
              RAD_E_UNSUPPORTED, "layout: codebook=mad on a mad encoding (as stored)");
    }
    RadEncoding st{};
    check(exl3i_encoding_make(64, EXL3I_CB_MAD, 1, 1024, 256, &st) == RAD_OK, "stacked encoding");
    check(embed_exl3i_layout(p.data(), (int)p.size(), EMBED_EXL3I_RECORD, &st, sel_rec, nullptr, 0, nullptr) !=
              RAD_E_UNSUPPORTED, "layout: a stacked weight is refused");
}

/* mad (the codebook the importer takes for token_embd too): small synthetic tables at the edge rates and both row
 * orders, the host gather against the oracle (one() checks every element bit for bit, rows read from the record) */
void mad_table() {
    const int ms[] = { 2, 16, 40, 128 };
    unsigned seed = 0x3AD;
    for (int m : ms)
        for (int r16 = 0; r16 < 2; ++r16) one(128, 128, m, EXL3I_CB_MAD, r16, seed++);
}

void refusals() {
    Table T = make(256, 128, 64, EXL3I_CB_MAD, 0, 7);
    std::vector<int32_t> tok = { 0, 1, 2 };
    std::vector<uint16_t> x(3 * 128);
    {
        Call c; bind(c, T, tok, x, "mad");
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_OK, "the well-formed call");
    }
    {
        Call c; bind(c, T, tok, x, "mad9");
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "unknown codebook");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.p[3] = RAD_INT("m", 63);
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "odd m");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.p[3] = RAD_INT("m", 130);
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "m past 128");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.p[2] = RAD_INT("n_vocab", 240);
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "n_vocab not in 128s");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.p[5] = RAD_INT("r16", 2);
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "r16 not 0 / 1");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.p[3] = RAD_INT("m", 72);
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "record of another rate");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[1].dtype = RAD_I32;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_DTYPE, "record not u32");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[2].dtype = RAD_BF16;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_DTYPE, "suh not f16");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[4].dtype = RAD_F16;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_DTYPE, "x not bf16");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[3].shape[0] = 128;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "svh too short");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[4].shape[0] = 2;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_SHAPE, "x has fewer rows than tokens");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.t[1].data = nullptr;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_INVAL, "no record");
    }
    {
        Call c; bind(c, T, tok, x, "mad"); c.a.n_t = 4;
        check(embed_exl3i_host_launch(&c.a, nullptr) == RAD_E_INVAL, "an operand short");
    }
}

}  // namespace

int main() {
    const int cbs[1] = { EXL3I_CB_MAD };
    const int ms[] = { 2, 8, 32, 36, 40, 64, 72, 104, 112, 128 };   /* K 0.125 0.5 2 2.25 2.5 4 4.5 6.5 7 8 */
    unsigned seed = 1;
    for (int m : ms)
        for (int cb : cbs)
            for (int r16 = 0; r16 < 2; ++r16) one(256, 256, m, cb, r16, seed++);
    mad_table();
    layout_hook();
    refusals();
    if (failures) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("embed_exl3i host: all checks passed\n");
    return 0;
}
