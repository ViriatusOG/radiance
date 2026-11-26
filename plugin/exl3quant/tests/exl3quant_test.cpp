/* exl3quant_test.cpp -- the `exl3i` quantiser (EXL3-V3's imports) through its plugin ABI, the way rad-convert drives
 * it: dlopen the built .so, take rad_quant_at(0), call encoding / row_block / quantize / decode. The fixture is built
 * here, in C++ (no python): an ourq-style level (fractional trellises in exllamav3's layout, stamp.json) for the mad
 * codebook, the mad and hash stamp forms, and the rows=r16 head. */
#include "rad_quant.h"
#include "exl3_decode.h"
#include "exl3i_encoding.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

int FAILS = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++FAILS;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

std::string DIR;   /* the fixture directory */


void put(const std::string& rel, const void* p, size_t n) {
    FILE* f = std::fopen((DIR + "/" + rel).c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", rel.c_str()); exit(2); }
    std::fwrite(p, 1, n, f);
    std::fclose(f);
}

void put_str(const std::string& rel, const std::string& s) { put(rel, s.data(), s.size()); }

/* One fixture tensor into the safetensors: its header entry (collected) and its bytes. */
struct ST {
    std::string header;   /* the JSON member */
    std::vector<char> data;
};

std::vector<ST> st_all;
uint64_t st_off = 0;   /* the file's data_offsets are relative to the end of the header */

void st_tensor(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape,
              const std::vector<char>& bytes) {
    ST t;
    std::string sh;
    for (size_t i = 0; i < shape.size(); ++i)
        sh += (i ? "," : "") + std::to_string(shape[i]);
    t.header = "\"" + name + "\": {\"dtype\": \"" + dtype + "\", \"shape\": [" + sh +
              + "], \"data_offsets\": [" + std::to_string(st_off) + ", " +
              std::to_string(st_off + bytes.size()) + "]}";
    t.data = bytes;
    st_all.push_back(t);
    st_off += bytes.size();
}

template <typename T>
void st_num(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape,
           const std::vector<T>& v) {
    std::vector<char> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    st_tensor(name, dtype, shape, b);
}

/* The options, with the string pointers fixed up. */
struct Opts {
    std::vector<RadParam> p;
    std::vector<std::string> keep;
    Opts& s(const char* k, const std::string& v) {
        keep.push_back(v);
        RadParam r{}; r.key = k; r.kind = RAD_P_STR; r.sval = nullptr;
        p.push_back(r);
        return *this;
    }
    Opts& i(const char* k, long long v) {
        RadParam r{}; r.key = k; r.kind = RAD_P_INT; r.ival = v;
        p.push_back(r);
        return *this;
    }
    const RadParam* data() {
        size_t j = 0;
        for (auto& r : p)
            if (r.kind == RAD_P_STR) r.sval = keep[j++].c_str();
        return p.data();
    }
    int n() const { return (int)p.size(); }
};

Opts base(const std::string& src) { return Opts().s("src", src); }

RadQuantWeight weight(const char* name, int64_t rows, int64_t cols) {
    RadQuantWeight w{};
    w.name = (char*)name;
    w.rows = rows;
    w.cols = cols;
    w.rank = 2;
    return w;
}

/* ------------------------------------------------------------------ exl3i (any rate, mad) */

const RadQuantizerInfo* QI = nullptr;
std::string DIRI;   /* an ourq-style level: fractional trellises, stamp.json codebook mad */

/* the level's stamp.json, its last run's codebook set (a level named for another codebook, for the stamp check) */
void stamp_codebook(const std::string& dir, const char* cb) {
    const std::string save = DIR;
    DIR = dir;
    put_str("stamp.json", std::string("[{\"stamp\": \"x\", \"args\": {\"codebook\": \"other\"}}, "
                                    "{\"stamp\": \"y\", \"args\": {\"codebook\": \"") + cb + "\"}}]");
    DIR = save;
}

/* A projection at rate m / 16: pseudo-random record words (any bit ring is a valid trellis) */
std::vector<char> proj_i(const std::string& base, int k, int n, int m, uint32_t seed) {
    std::vector<char> tr((size_t)(k / 16) * (n / 16) * m * 2);
    uint32_t x = seed;
    for (size_t i = 0; i + 4 <= tr.size(); i += 4) { x = x * 1664525u + 1013904223u; std::memcpy(tr.data() + i, &x, 4); }
    st_tensor(base + ".trellis", "I16", {k / 16, n / 16, m}, tr);
    std::vector<int16_t> suh(k), svh(n);
    for (int i = 0; i < k; ++i) suh[i] = (int16_t)(0x3C00 + (i % 64));   /* ~1.0 */
    for (int i = 0; i < n; ++i) svh[i] = (int16_t)(0x3C00 - (i % 32));
    st_num(base + ".suh", "F16", {k}, suh);
    st_num(base + ".svh", "F16", {n}, svh);
    return tr;
}

std::vector<char> TR_DOWN, TR_GATE, TR_UP, TR_AB;

void build_fixture_i() {
    st_all.clear(); st_off = 0;
    DIRI = std::tmpnam(nullptr);
    std::string save = DIR; DIR = DIRI;
    if (std::system(("mkdir -p '" + DIRI + "' 2>&1").c_str()) != 0) { std::fprintf(stderr, "mkdir\n"); exit(2); }
    put_str("config.json", "{ \"quantization_config\": { \"quant_method\": \"exl3\" }, \"text_config\": { \"num_hidden_layers\": 2 } }");
    put_str("stamp.json", "[{\"stamp\": \"x\", \"args\": {\"codebook\": \"other\"}}, {\"stamp\": \"y\", \"args\": {\"codebook\": \"mad\"}}]");
    TR_DOWN = proj_i("model.language_model.layers.0.mlp.down_proj", 256, 128, 40, 1);     /* K2.5 */
    TR_GATE = proj_i("model.language_model.layers.0.mlp.gate_proj", 128, 256, 36, 2);     /* K2.25 */
    TR_UP   = proj_i("model.language_model.layers.0.mlp.up_proj",   128, 256, 36, 3);
    proj_i("model.language_model.layers.0.self_attn.q_proj", 128, 128, 2, 4);           /* K0.125 */
    proj_i("model.language_model.layers.0.self_attn.k_proj", 128, 128, 128, 5);           /* K8 */
    proj_i("model.language_model.layers.0.self_attn.v_proj", 128, 128, 37, 6);           /* odd: no rate */
    proj_i("model.language_model.layers.0.linear_attn.in_proj_z", 128, 128, 40, 7);               /* no marker file */
    proj_i("model.language_model.layers.1.mlp.gate_proj", 128, 256, 36, 8);               /* parts disagree */
    proj_i("model.language_model.layers.1.mlp.up_proj",   128, 256, 40, 9);
    TR_AB = proj_i("model.language_model.layers.1.linear_attn.in_proj_ab", 256, 128, 64, 10);   /* a | b, K4 */
    std::string h = "{\"__metadata__\": {\"format\": \"pt\"}";
    for (auto& t : st_all) h += "," + t.header;
    h += "}";
    std::string body(8, (char)0);
    for (int i = 0; i < 8; ++i) body[i] = (char)((uint64_t)h.size() >> (8 * i));
    body += h;
    for (auto& t : st_all) body.append(t.data.data(), t.data.size());
    put("model.safetensors", body.data(), body.size());
    DIR = save;
}

int enc_i(const std::string& name, int64_t rows, int64_t cols, Opts& o, RadEncoding* e) {
    RadQuantWeight w = weight(name.c_str(), rows, cols);
    return QI->encoding(o.data(), o.n(), &w, e);
}

/* quantize into planes sized from the encoding; plane[i] the bytes */
int quant_i(const RadEncoding& e, const std::string& name, int64_t rows, int64_t cols, Opts& o,
            std::vector<std::vector<char>>* plane) {
    plane->assign(e.n_planes, {});
    std::vector<void*> ptr(e.n_planes);
    for (int i = 0; i < e.n_planes; ++i) {
        (*plane)[i].assign((size_t)rad_enc_plane_bytes(&e.plane[i], rows, cols), 0);
        ptr[i] = (*plane)[i].data();
    }
    RadQuantWeight w = weight(name.c_str(), rows, cols);
    return QI->quantize(o.data(), o.n(), &w, nullptr, 0, rows, ptr.data());
}

/* W_hat straight from an exllamav3 trellis [k/16][n/16][m] (tile order as stored), radiance's [n][k] view */
std::vector<float> what_from_source(const std::vector<char>& tr, int k, int n, int m, int cb) {
    std::vector<float> w((size_t)n * k);
    const int kt = k / 16, nt = n / 16;
    uint16_t val[256];
    for (int tk = 0; tk < kt; ++tk)
        for (int tn = 0; tn < nt; ++tn) {
            std::vector<uint32_t> words(m / 2);
            std::memcpy(words.data(), tr.data() + ((size_t)tk * nt + tn) * m * 2, (size_t)m * 2);
            exl3i_decode_tile(words.data(), m, cb, val, nullptr);
            for (int p = 0; p < 256; ++p) {
                int r, c;
                exl3_pos_to_tile(p, &r, &c);
                w[(size_t)(tn * 16 + c) * k + tk * 16 + r] = exl3_half_to_float(val[p]);
            }
        }
    return w;
}

void test_exl3i() {
    build_fixture_i();
    Opts o = base(DIRI);
    o.s("codebook", "mad");
    RadEncoding e;
    int m = 0, cb = 0, stk = 0;
    int64_t r = 0, c = 0;
    char ts[64];

    /* encodings */
    check(enc_i("blk.0.ffn_down.weight", 128, 256, o, &e) == RAD_OK, "exl3i: down encodes");
    check(exl3i_encoding_read(&e, &m, &cb, &stk, &r, &c) == RAD_OK && m == 40 && cb == EXL3I_CB_MAD && !stk &&
          r == 128 && c == 256, "exl3i: down is K2.5 mad [128, 256]");
    check(!std::strcmp(e.transform, "mad/h128/k2.5"), "exl3i: transform spelling");
    check(rad_enc_valid(&e) == 1, "exl3i: the encoding is valid for the core");
    std::snprintf(ts, sizeof ts, "%lld", (long long)e.plane[EXL3I_PLANE_RECORD].extent[1]);
    check(e.plane[EXL3I_PLANE_RECORD].extent[0] == 8 && e.plane[EXL3I_PLANE_RECORD].extent[1] == 16 * 20,
          "exl3i: record table [rows/16, (cols/16) m/2]");
    check(enc_i("blk.0.ffn_gate_up.weight", 512, 128, o, &e) == RAD_OK &&
          exl3i_encoding_read(&e, &m, &cb, &stk, &r, &c) == RAD_OK && m == 36 && stk == 1, "exl3i: gate_up stacked K2.25");
    check(enc_i("blk.0.attn_qg.weight", 128, 128, o, &e) == RAD_OK &&
          exl3i_encoding_read(&e, &m, nullptr, nullptr, nullptr, nullptr) == RAD_OK && m == 2 &&
          !std::strcmp(e.transform, "mad/h128/k0.125"), "exl3i: sub-1 K0.125");
    check(enc_i("blk.0.attn_k.weight", 128, 128, o, &e) == RAD_OK &&
          exl3i_encoding_read(&e, &m, nullptr, nullptr, nullptr, nullptr) == RAD_OK && m == 128, "exl3i: integer K8");

    /* refusals */
    check(enc_i("blk.0.attn_v.weight", 128, 128, o, &e) == RAD_E_FORMAT, "exl3i: an odd 16 K (37) is E_FORMAT");
    check(enc_i("blk.0.ssm_z.weight", 128, 128, o, &e) == RAD_OK, "exl3i: a trellis without the old marker file encodes (the marker is not read)");
    check(enc_i("blk.1.ffn_gate_up.weight", 512, 128, o, &e) == RAD_E_DTYPE, "exl3i: stacked parts at two rates");
    Opts nocb = base(DIRI);
    check(enc_i("blk.0.ffn_down.weight", 128, 256, nocb, &e) == RAD_E_INVAL, "exl3i: codebook= is required");
    Opts plain = base(DIRI);
    plain.s("codebook", "mad");
    stamp_codebook(DIRI, "other");   /* a level that named another codebook refuses mad */
    check(enc_i("blk.0.ffn_down.weight", 128, 256, plain, &e) == RAD_E_DTYPE, "exl3i: codebook=mad against a stamp that names another codebook is E_DTYPE");
    stamp_codebook(DIRI, "mad");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, plain, &e) == RAD_OK, "exl3i: codebook=mad against the mad stamp");
    Opts r25 = base(DIRI), r225 = base(DIRI), rbad = base(DIRI);
    r25.s("codebook", "mad").s("rate", "2.5");
    r225.s("codebook", "mad").s("rate", "2.25");
    rbad.s("codebook", "mad").s("rate", "2.50");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, r25, &e) == RAD_OK, "exl3i: rate=2.5 on the K2.5 tensor");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, r225, &e) == RAD_E_DTYPE, "exl3i: rate=2.25 on K2.5 is E_DTYPE");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, rbad, &e) == RAD_E_INVAL, "exl3i: rate=2.50 (not canonical) is E_INVAL");
    Opts xc = base(DIRI);
    xc.s("codebook", "other");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, xc, &e) == RAD_E_INVAL, "exl3i: a codebook other than mad is E_INVAL (the option names no codebook)");
    check(enc_i("blk.0.ffn_down.weight", 100, 256, o, &e) == RAD_E_SHAPE, "exl3i: rows not in 128s");

    /* the import: down, record rows in k order, scales copied; the decode against the source's tiles */
    {
        check(enc_i("blk.0.ffn_down.weight", 128, 256, o, &e) == RAD_OK, "exl3i: down encodes (2)");
        std::vector<std::vector<char>> pl;
        check(quant_i(e, "blk.0.ffn_down.weight", 128, 256, o, &pl) == RAD_OK, "exl3i: down imports");
        bool zero = true;
        for (char b : pl[EXL3I_PLANE_CODES]) zero = zero && !b;
        check(zero, "exl3i: the codes plane is zero");
        const auto ref = what_from_source(TR_DOWN, 256, 128, 40, EXL3I_CB_MAD);
        Exl3iView v{};
        v.rows = 128; v.cols = 256; v.m = 40; v.cb = EXL3I_CB_MAD;
        v.record = reinterpret_cast<const uint32_t*>(pl[EXL3I_PLANE_RECORD].data());
        std::vector<float> got((size_t)128 * 256);
        check(exl3i_decode_what_rows(&v, 0, 128, got.data()) == 0 && got == ref,
              "exl3i: W_hat from the record = W_hat from the source's tiles");
        uint16_t s0 = 0, v0 = 0;
        std::memcpy(&s0, pl[EXL3I_PLANE_SUH].data() + 2 * 5, 2);
        std::memcpy(&v0, pl[EXL3I_PLANE_SVH].data() + 2 * 7, 2);
        check(s0 == 0x3C05 && v0 == 0x3C00 - 7, "exl3i: suh / svh are the source's");
        std::vector<const void*> cp;
        for (auto& p : pl) cp.push_back(p.data());
        RadQuantWeight w = weight("blk.0.ffn_down.weight", 128, 256);
        std::vector<float> full((size_t)128 * 256), want((size_t)128 * 256);
        v.suh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SUH].data());
        v.svh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SVH].data());
        check(QI->decode(&e, &w, cp.data(), 0, 128, full.data()) == RAD_OK, "exl3i: the decode hook decodes");
        check(exl3i_decode_rows(&v, 0, 128, want.data()) == 0 && full == want, "exl3i: decode hook = the oracle");
        RadEncoding ex;
        rad_enc_clear(&ex);
        rad_enc_copy_str(ex.scheme, "exl3");
        check(QI->decode(&ex, &w, cp.data(), 0, 128, full.data()) == RAD_E_UNSUPPORTED,
              "exl3i: the decode hook declines an exl3 encoding");
    }
    /* stacked: gate rows, then up rows; suh / suh.1 */
    {
        check(enc_i("blk.0.ffn_gate_up.weight", 512, 128, o, &e) == RAD_OK, "exl3i: gate_up encodes (2)");
        std::vector<std::vector<char>> pl;
        check(quant_i(e, "blk.0.ffn_gate_up.weight", 512, 128, o, &pl) == RAD_OK, "exl3i: gate_up imports");
        const auto g = what_from_source(TR_GATE, 128, 256, 36, EXL3I_CB_MAD);
        const auto u = what_from_source(TR_UP, 128, 256, 36, EXL3I_CB_MAD);
        Exl3iView v{};
        v.rows = 512; v.cols = 128; v.m = 36; v.cb = EXL3I_CB_MAD;
        v.record = reinterpret_cast<const uint32_t*>(pl[EXL3I_PLANE_RECORD].data());
        std::vector<float> got((size_t)512 * 128);
        check(exl3i_decode_what_rows(&v, 0, 512, got.data()) == 0, "exl3i: gate_up W_hat decodes");
        check(std::vector<float>(got.begin(), got.begin() + 256 * 128) == g &&
              std::vector<float>(got.begin() + 256 * 128, got.end()) == u, "exl3i: gate rows then up rows");
        check(pl.size() == 5 && pl[EXL3I_PLANE_SUH1].size() == 256, "exl3i: suh.1 present");
    }
    /* the GDN's fused a | b : ssm_ab imports the quantizer's in_proj_ab key, one trellis */
    {
        check(enc_i("blk.1.ssm_ab.weight", 128, 256, o, &e) == RAD_OK &&
              exl3i_encoding_read(&e, &m, &cb, &stk, &r, &c) == RAD_OK && m == 64 && cb == EXL3I_CB_MAD && !stk &&
              r == 128 && c == 256, "exl3i: ssm_ab is K4 mad [128, 256] from in_proj_ab");
        std::vector<std::vector<char>> pl;
        check(quant_i(e, "blk.1.ssm_ab.weight", 128, 256, o, &pl) == RAD_OK, "exl3i: ssm_ab imports");
        Exl3iView v{};
        v.rows = 128; v.cols = 256; v.m = 64; v.cb = EXL3I_CB_MAD;
        v.record = reinterpret_cast<const uint32_t*>(pl[EXL3I_PLANE_RECORD].data());
        std::vector<float> got((size_t)128 * 256);
        check(exl3i_decode_what_rows(&v, 0, 128, got.data()) == 0 &&
              got == what_from_source(TR_AB, 256, 128, 64, EXL3I_CB_MAD), "exl3i: ssm_ab W_hat = the source's tiles");
        check(enc_i("blk.0.ssm_ab.weight", 128, 256, o, &e) != RAD_OK, "exl3i: a layer without in_proj_ab is refused");
    }
}

/* ------------------------------------------------------------------ the level builder (mad's and the i4 stamp's sources) */

std::vector<char> TR4_DOWN, TR4_GATE, TR4_UP, TR4_O;

/* a one-file level in `dir` with the given stamp.json; fills the trellises it is asked for */
void build_level_i4(std::string* dir, const char* stamp, bool full) {
    st_all.clear(); st_off = 0;
    *dir = std::tmpnam(nullptr);
    std::string save = DIR; DIR = *dir;
    if (std::system(("mkdir -p '" + *dir + "' 2>&1").c_str()) != 0) { std::fprintf(stderr, "mkdir\n"); exit(2); }
    put_str("config.json", "{ \"quantization_config\": { \"quant_method\": \"exl3\" }, \"text_config\": { \"num_hidden_layers\": 1 } }");
    put_str("stamp.json", stamp);
    TR4_DOWN = proj_i("model.language_model.layers.0.mlp.down_proj", 256, 128, 32, 11);    /* K2 */
    if (full) {
        TR4_GATE = proj_i("model.language_model.layers.0.mlp.gate_proj", 128, 256, 48, 12);  /* K3 */
        TR4_UP   = proj_i("model.language_model.layers.0.mlp.up_proj",   128, 256, 48, 13);
        TR4_O    = proj_i("model.language_model.layers.0.self_attn.o_proj", 128, 128, 8, 14);  /* K0.5 */
    }
    std::string h = "{\"__metadata__\": {\"format\": \"pt\"}";
    for (auto& t : st_all) h += "," + t.header;
    h += "}";
    std::string body(8, (char)0);
    for (int i = 0; i < 8; ++i) body[i] = (char)((uint64_t)h.size() >> (8 * i));
    body += h;
    for (auto& t : st_all) body.append(t.data.data(), t.data.size());
    put("model.safetensors", body.data(), body.size());
    DIR = save;
}

/* ------------------------------------------------------------------ exl3i, codebook mad  */

#define EXL3I_A8_WS_TEST 0.027069091796875f   /* 4 K_INV (fp16 0x1EEE), exact */
std::string DIRM[6];   /* mad / hash levels, one a stamp form (good and bad constants) */

void test_exl3i_mad() {
    static const char* const stamps[6] = {
        "[{\"stamp\": \"r\", \"args\": {\"codebook\": \"mad\", \"mad\": {\"a\": \"0xd0474b\", \"b\": \"0x2545f491\"}}}]",   /* rxq */
        "[{\"stamp\": \"o\", \"args\": {\"codebook\": \"mad\", \"mad_a\": 13649739, \"mad_b\": 625341585}}]",          /* ourq */
        "[{\"stamp\": \"h\", \"args\": {\"codebook\": \"hash\", \"hash\": \"d0474b-2545f491\"}}]",
        "[{\"stamp\": \"x\", \"args\": {\"codebook\": \"mad\", \"mad\": {\"a\": \"0xb5297b\", \"b\": \"0x2545f491\"}}}]",   /* other a */
        "[{\"stamp\": \"y\", \"args\": {\"codebook\": \"mad\", \"mad_a\": 13649739, \"mad_b\": 1}}]",                  /* other b */
        "[{\"stamp\": \"z\", \"args\": {\"codebook\": \"hash\", \"hash\": \"d0474b-2545f491-m1bdfe861\"}}]",            /* a map */
    };
    for (int i = 0; i < 6; ++i) build_level_i4(&DIRM[i], stamps[i], i == 0);
    RadEncoding e;
    int m = 0, cb = 0;
    for (int i = 0; i < 6; ++i) {
        Opts o = base(DIRM[i]);
        o.s("codebook", "mad");
        const int rc = enc_i("blk.0.ffn_down.weight", 128, 256, o, &e);
        char msg[96];
        std::snprintf(msg, sizeof msg, "exl3i mad: stamp form %d %s", i, i < 3 ? "accepted" : "refused (another code)");
        check(i < 3 ? rc == RAD_OK && exl3i_encoding_read(&e, &m, &cb, nullptr, nullptr, nullptr) == RAD_OK &&
                          cb == EXL3I_CB_MAD && m == 32 && !std::strcmp(e.transform, "mad/h128/k2")
                    : rc == RAD_E_DTYPE, msg);
        Opts o8 = base(DIRM[i]);
        o8.s("codebook", "other");
        std::snprintf(msg, sizeof msg, "exl3i mad: stamp form %d refuses a codebook other than mad (E_INVAL)", i);
        check(enc_i("blk.0.ffn_down.weight", 128, 256, o8, &e) == RAD_E_INVAL, msg);
    }
    /* the import: W_hat = the source's tiles read in mad, every value on the int8 grid; decode hook = the oracle */
    Opts o = base(DIRM[0]);
    o.s("codebook", "mad");
    check(enc_i("blk.0.ffn_gate_up.weight", 512, 128, o, &e) == RAD_OK && !std::strcmp(e.transform, "mad/h128/k3"),
          "exl3i mad: gate_up stacked K3 (mad/h128/k3)");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, o, &e) == RAD_OK, "exl3i mad: down encodes (2)");
    std::vector<std::vector<char>> pl;
    check(quant_i(e, "blk.0.ffn_down.weight", 128, 256, o, &pl) == RAD_OK, "exl3i mad: down imports");
    const auto ref = what_from_source(TR4_DOWN, 256, 128, 32, EXL3I_CB_MAD);
    Exl3iView v{};
    v.rows = 128; v.cols = 256; v.m = 32; v.cb = EXL3I_CB_MAD;
    v.record = reinterpret_cast<const uint32_t*>(pl[EXL3I_PLANE_RECORD].data());
    std::vector<float> got((size_t)128 * 256);
    bool grid = true;
    check(exl3i_decode_what_rows(&v, 0, 128, got.data()) == 0 && got == ref, "exl3i mad: W_hat from the record = the source's tiles");
    for (float x : got) {   /* x = fp16(q 4 K_INV) for an integer q, |q| <= 126 */
        const float q = std::nearbyint(x / EXL3I_A8_WS_TEST);
        grid = grid && std::fabs(q) <= 126.0f && (float)(_Float16)(q * EXL3I_A8_WS_TEST) == x;
    }
    check(grid, "exl3i mad: every W_hat value is fp16(q 4 K_INV), |q| <= 126");
    std::vector<const void*> cp;
    for (auto& p : pl) cp.push_back(p.data());
    RadQuantWeight w = weight("blk.0.ffn_down.weight", 128, 256);
    std::vector<float> full((size_t)128 * 256), want((size_t)128 * 256);
    v.suh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SUH].data());
    v.svh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SVH].data());
    check(QI->decode(&e, &w, cp.data(), 0, 128, full.data()) == RAD_OK, "exl3i mad: the decode hook decodes");
    check(exl3i_decode_rows(&v, 0, 128, want.data()) == 0 && full == want, "exl3i mad: decode hook = the oracle");
}

/* ------------------------------------------------------------------ exl3i, rows=r16 (the head in R16 order) */

std::string DIRR, DIRR_BAD;   /* ourq head levels stamped with head_perm: the R16 file / an identity file */

void put_npy_i64(const std::string& path, const std::vector<int64_t>& v) {
    std::string d = "{'descr': '<i8', 'fortran_order': False, 'shape': (" + std::to_string(v.size()) + ",), }";
    while ((10 + d.size() + 1) % 64) d += ' ';
    d += '\n';
    std::string f = std::string("\x93NUMPY\x01\x00", 8);
    f += (char)(d.size() & 0xFF); f += (char)(d.size() >> 8);
    f += d;
    f.append(reinterpret_cast<const char*>(v.data()), v.size() * 8);
    FILE* o = std::fopen(path.c_str(), "wb");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    std::fwrite(f.data(), 1, f.size(), o);
    std::fclose(o);
}

/* a head level: lm_head [256, 128] at K4, stamp.json args.head_perm = dir/R16.npy, written as `perm` */
std::vector<char> TRH;
void build_level_head(std::string* dir, const std::vector<int64_t>& perm, bool npy) {
    st_all.clear(); st_off = 0;
    *dir = std::tmpnam(nullptr);
    std::string save = DIR; DIR = *dir;
    if (std::system(("mkdir -p '" + *dir + "' 2>&1").c_str()) != 0) { std::fprintf(stderr, "mkdir\n"); exit(2); }
    put_str("config.json", "{ \"quantization_config\": { \"quant_method\": \"exl3\" }, \"text_config\": { \"num_hidden_layers\": 1 } }");
    put_str("stamp.json", "[{\"stamp\": \"h\", \"args\": {\"codebook\": \"mad\", \"head_perm\": \"" + *dir + "/R16.npy\"}}]");
    if (npy) put_npy_i64(*dir + "/R16.npy", perm);
    TRH = proj_i("lm_head", 128, 256, 64, 21);
    std::string h = "{\"__metadata__\": {\"format\": \"pt\"}";
    for (auto& t : st_all) h += "," + t.header;
    h += "}";
    std::string body(8, (char)0);
    for (int i = 0; i < 8; ++i) body[i] = (char)((uint64_t)h.size() >> (8 * i));
    body += h;
    for (auto& t : st_all) body.append(t.data.data(), t.data.size());
    put("model.safetensors", body.data(), body.size());
    DIR = save;
}

void test_exl3i_r16() {
    const int64_t N = 256, K = 128;
    std::vector<int64_t> r16(N), ident(N);
    for (int64_t s = 0; s < N; ++s) { r16[s] = exl3i_r16_row(s, N); ident[s] = s; }
    build_level_head(&DIRR_BAD, ident, true);
    build_level_head(&DIRR, r16, true);   /* last: TRH is this level's trellis */
    Opts o = base(DIRR), nat = base(DIRR), bad = base(DIRR_BAD), plain = base(DIRI), junk = base(DIRR);
    o.s("codebook", "mad").s("rows", "r16");
    nat.s("codebook", "mad");
    bad.s("codebook", "mad").s("rows", "r16");
    plain.s("codebook", "mad").s("rows", "r16");
    junk.s("codebook", "mad").s("rows", "r8");
    RadEncoding e;
    int r = -1, m = 0;
    check(enc_i("output.weight", N, K, o, &e) == RAD_OK && exl3i_encoding_read_r16(&e, &m, 0, 0, 0, 0, &r) == RAD_OK && r == 1 &&
          m == 64 && !std::strcmp(e.transform, "mad/h128/k4/r16"), "exl3i r16: the head encodes as mad/h128/k4/r16");
    check(enc_i("output.weight", N, K, nat, &e) == RAD_E_DTYPE, "exl3i r16: a head_perm level without rows=r16 is refused");
    check(enc_i("output.weight", N, K, bad, &e) == RAD_E_DTYPE, "exl3i r16: a head_perm file that is not R16 is refused");
    check(enc_i("blk.0.ffn_down.weight", 128, 256, plain, &e) == RAD_E_DTYPE, "exl3i r16: rows=r16 without a head_perm is refused");
    check(enc_i("output.weight", N, K, junk, &e) == RAD_E_INVAL, "exl3i r16: rows=r8 is E_INVAL");
    /* the import: the record in stored (R16) order, unchanged; the decode hook answers vocabulary rows */
    check(enc_i("output.weight", N, K, o, &e) == RAD_OK, "exl3i r16: encodes (2)");
    std::vector<std::vector<char>> pl;
    check(quant_i(e, "output.weight", N, K, o, &pl) == RAD_OK, "exl3i r16: imports");
    Exl3iView v{};
    v.rows = N; v.cols = K; v.m = 64; v.cb = EXL3I_CB_MAD;
    v.record = reinterpret_cast<const uint32_t*>(pl[EXL3I_PLANE_RECORD].data());
    v.suh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SUH].data());
    v.svh = reinterpret_cast<const uint16_t*>(pl[EXL3I_PLANE_SVH].data());
    std::vector<float> stored((size_t)N * K), logical((size_t)N * K), part((size_t)40 * K);
    check(exl3i_decode_rows(&v, 0, N, stored.data()) == 0, "exl3i r16: the stored rows decode");
    std::vector<const void*> cp;
    for (auto& p : pl) cp.push_back(p.data());
    RadQuantWeight w = weight("output.weight", N, K);
    check(QI->decode(&e, &w, cp.data(), 0, N, logical.data()) == RAD_OK, "exl3i r16: the decode hook decodes");
    bool ok = true;
    for (int64_t s = 0; s < N; ++s) ok = ok && !std::memcmp(&logical[(size_t)exl3i_r16_row(s, N) * K], &stored[(size_t)s * K], K * 4);
    check(ok, "exl3i r16: decode hook row exl3i_r16_row(s) = stored row s, every s");
    check(QI->decode(&e, &w, cp.data(), 100, 40, part.data()) == RAD_OK &&
          !std::memcmp(part.data(), &logical[(size_t)100 * K], part.size() * 4), "exl3i r16: a row range of the decode hook");
    for (const std::string* d : {&DIRR, &DIRR_BAD}) std::remove((*d + "/R16.npy").c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: exl3quant_test <exl3quant.so>\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto ver = (uint32_t (*)())dlsym(h, "rad_plugin_abi_version");
    auto inf = (const RadPluginInfo* (*)())dlsym(h, "rad_plugin_info");
    auto nq = (int (*)())dlsym(h, "rad_quant_count");
    auto at = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    check(ver && inf && nq && at, "the plugin's ABI entry points");
    check(ver() == RAD_ABI_VERSION, "the plugin's ABI matches this radiance");
    const RadPluginInfo* pi = inf();
    check(pi && pi->kind == RAD_PLUGIN_QUANT && !std::strcmp(pi->name, "exl3quant"), "it is the quant plugin");
    check(nq() == 1, "one quantiser");
    QI = at(0);
    check(QI && !std::strcmp(QI->name, "exl3i") && QI->decode, "rad_quant_at(0) is exl3i, with a decoder");
    check(at(1) == nullptr, "no second quantiser");

    test_exl3i();
    test_exl3i_mad();
    test_exl3i_r16();

    for (const std::string* d : {&DIRI, &DIRR, &DIRR_BAD, &DIRM[0], &DIRM[1], &DIRM[2], &DIRM[3], &DIRM[4], &DIRM[5]}) {
        std::remove((*d + "/config.json").c_str());
        std::remove((*d + "/stamp.json").c_str());
        std::remove((*d + "/model.safetensors").c_str());
        ::rmdir(d->c_str());
    }
    std::fprintf(stderr, "%s: %d failure(s)\n", FAILS ? "FAIL" : "ok", FAILS);
    return FAILS ? 1 : 0;
}
