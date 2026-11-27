/* exl3_quant.cpp -- the `exl3i` quantiser: IMPORTS EXL3-V3 trellises (any rate m / 16 in 1/8 steps,
 * codebook mad) from one exllamav3-layout level directory (src=DIR) into the exl3i encoding
 * (exl3import/include/exl3i_encoding.h) for the qwen35_exl3 architecture of this tree. Nothing is
 * quantised here: the trellis is the level's (the EXL3-V3 quantizer writes it).
 *
 * A name that is not one of the mapped weights is DECLINED (RAD_E_UNSUPPORTED) and the next recipe
 * rule decides it.
 *
 * WHICH NAMES (the qwen35_exl3 architecture of this tree):
 *   blk.<L>.attn_qg.weight    ->  self_attn.q_proj            (query AND gate, one tensor)
 *   blk.<L>.attn_k.weight     ->  self_attn.k_proj
 *   blk.<L>.attn_v.weight     ->  self_attn.v_proj
 *   blk.<L>.attn_output.weight->  self_attn.o_proj
 *   blk.<L>.ffn_gate.weight   ->  mlp.gate_proj
 *   blk.<L>.ffn_up.weight     ->  mlp.up_proj
 *   blk.<L>.ffn_down.weight   ->  mlp.down_proj
 *   blk.<L>.ssm_qkv.weight    ->  linear_attn.in_proj_qkv
 *   blk.<L>.ssm_z.weight      ->  linear_attn.in_proj_z
 *   blk.<L>.ssm_out.weight    ->  linear_attn.out_proj
 *   blk.<L>.ssm_ab.weight     ->  linear_attn.in_proj_ab      (a | b | padding to 128 rows, as one
 *                                                              trellis: the quantizer's fused key;
 *                                                              qwen35_exl3 maps it)
 *   output.weight             ->  lm_head                     (top-level; the embed fallback is
 *                                                              plain)
 *   token_embd.weight         ->  model.language_model.embed_tokens   (exl3i only: a trellis token
 *                                 table, served by exl3i's embed_exl3i from the stored record)
 *   gemma4_exl3 (Gemma 4 dense) shares the checkpoint's layer prefix and the ffn / k / v names; it
 *   adds
 *   blk.<L>.attn_q.weight     ->  self_attn.q_proj            (q alone: Gemma 4 has no output gate)
 *   blk.<L>.attn_o.weight     ->  self_attn.o_proj
 *   blk.<L>.ffn_gate_up.weight stacks mlp.gate_proj | mlp.up_proj as for Qwen's MLP. A Gemma 4
 *   global layer (k == v) has no v_proj, hence no blk.<L>.attn_v.weight.
 *   blk.<L>.ssm_in.weight is REFUSED by name: 10416 rows at this model is not in 128s, so the
 *   architecture serves ssm_qkv exl3 beside plain a/b (the GDN ops' optional gate/beta operands).
 * The MoE forms (blk.<L>.ffn_*_exps.<E>.weight) are kept for the qwen4exp architecture.
 */
#include "rad_quant.h"
#include "rad_abi_layout.h"
#include "exl3_decode.h"
#include "exl3_src.h"
#include "exl3i_encoding.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void say(const char* weight, const std::string& why) {
    std::fprintf(stderr, "radiance: exl3: '%s': %s\n", weight ? weight : "?", why.c_str());
}

/* A parsed weight name. The MoE forms (blk.<L>.ffn_gate_up_exps.<E>.weight and
 * blk.<L>.ffn_down_exps.<E>.weight) stay kMoE: recognised, then declined. */
struct Name {
    enum Kind { kMoE, kDense, kMtp, kHead, kEmbed } kind = kMoE;
    int layer = -1;
    long long expert = -1;
    bool stacked = false;
    bool fused = false;    /* ssm_in: recognised, refused by name in front_i() */
    const char* part[2] = { nullptr, nullptr };   /* dense: the checkpoint projection(s) */
};

struct DenseRule {
    const char* name;
    const char* part[2];
    bool stacked;
};

const DenseRule kDense[] = {
    { "attn_qg", { "self_attn.q_proj", nullptr }, false },
    { "attn_q", { "self_attn.q_proj", nullptr }, false },   /* gemma4_exl3: q alone (no gate) */
    { "attn_o", { "self_attn.o_proj", nullptr }, false },   /* ...and its o_proj name */
    { "attn_k", { "self_attn.k_proj", nullptr }, false },
    { "attn_v", { "self_attn.v_proj", nullptr }, false },
    { "attn_out", { "self_attn.o_proj", nullptr }, false },
    { "attn_output", { "self_attn.o_proj", nullptr }, false },
    { "ffn_gate_up", { "mlp.gate_proj", "mlp.up_proj" }, true },
    { "ffn_gate", { "mlp.gate_proj", nullptr }, false },   /* qwen35_exl3's name */
    { "ffn_up", { "mlp.up_proj", nullptr }, false },
    { "ffn_down", { "mlp.down_proj", nullptr }, false },
    { "ssm_qkv", { "linear_attn.in_proj_qkv", nullptr }, false },
    { "ssm_inz", { "linear_attn.in_proj_qkv", nullptr }, false },   /* legacy: qkv only */
    { "ssm_z", { "linear_attn.in_proj_z", nullptr }, false },
    { "ssm_out", { "linear_attn.out_proj", nullptr }, false },
    { "ssm_ab", { "linear_attn.in_proj_ab", nullptr }, false },   /* a | b | padding, one trellis */
};

const DenseRule kMtpRules[] = {
    { "mlp_gate_up", { "mlp.gate_proj", "mlp.up_proj" }, true },
    { "mlp_down", { "mlp.down_proj", nullptr }, false },
};

void take(Name* out, Name::Kind kind, const DenseRule& r) {
    out->kind = kind;
    out->stacked = r.stacked;
    out->part[0] = r.part[0];
    out->part[1] = r.part[1];
}

bool parse_name(const char* s, Name* out) {
    int at = -1;
    if (!s) return false;
    if (std::strcmp(s, "output.weight") == 0) {
        out->kind = Name::kHead;
        out->part[0] = "lm_head";
        return true;
    }
    if (std::strcmp(s, "token_embd.weight") == 0) {
        out->kind = Name::kEmbed;
        out->part[0] = "model.language_model.embed_tokens";
        return true;
    }
    if (std::sscanf(s, "blk.%d.ffn_gate_up_exps.%lld.weight%n", &out->layer, &out->expert,
                    &at) == 2 &&
        at > 0 && s[at] == 0) {
        out->stacked = true;
        return out->layer >= 0 && out->expert >= 0;
    }
    if (at = -1, std::sscanf(s, "blk.%d.ffn_down_exps.%lld.weight%n", &out->layer, &out->expert,
                             &at) == 2 &&
        at > 0 && s[at] == 0)
        return out->layer >= 0 && out->expert >= 0;
    char part[4104];
    if (at = -1, std::sscanf(s, "blk.%d.ssm_in.weight%n", &out->layer, &at) == 1 && at > 0 &&
        s[at] == 0)
        return (out->kind = Name::kDense, out->fused = true, out->layer >= 0);
    if (at = -1, std::sscanf(s, "blk.%d.%4095[^.\n\t]%n", &out->layer, part, &at) == 2 && at > 0 &&
        std::strcmp(s + at, ".weight") == 0)
        for (const auto& r : kDense) {
            if (std::strcmp(part, r.name)) continue;
            take(out, Name::kDense, r);
            return out->layer >= 0;
        }
    if (at = -1, std::sscanf(s, "mtp.%4095[^.\n\t]%n", part, &at) == 1 && at > 0 &&
        std::strcmp(s + at, ".weight") == 0)
        for (const auto& r : kMtpRules) {
            if (std::strcmp(part, r.name)) continue;
            take(out, Name::kMtp, r);
            return true;
        }
    return false;
}

/* The prefix a dense/mtp weight's projections live under, and its source layer bound. */
int prefix_for(const char* weight, const Name& nm, exl3::Source* s, char* prefix, size_t n) {
    if (nm.kind == Name::kHead || nm.kind == Name::kEmbed)
        prefix[0] = 0;   /* top-level tensors: lm_head and the full embed_tokens name */
    else if (nm.kind == Name::kMtp) std::snprintf(prefix, n, "mtp.layers.0");
    else if (nm.layer < s->n_layers())
        std::snprintf(prefix, n, "model.language_model.layers.%d", nm.layer);
    else {
        say(weight, "layer " + std::to_string(nm.layer) + " is past the source's " +
                    std::to_string(s->n_layers()) + " layers");
        return RAD_E_NOTFOUND;
    }
    return RAD_OK;
}

int64_t row_block(const RadParam*, int, const RadQuantWeight*) { return 0; }

/* One tensor, checked for dtype and shape, read into dst. */
int fetch(exl3::Source* s, const std::string& name, const char* dtype,
          const std::vector<int64_t>& shape, void* dst, std::string* why) {
    exl3::Tensor t;
    const int f = s->find(name, &t, why);
    if (f < 0) return RAD_E_IO;
    if (f == 0) { *why = name + " is not in the source"; return RAD_E_NOTFOUND; }
    if (t.dtype != dtype || t.shape != shape) {
        std::string want, got;
        for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
        for (int64_t d : t.shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
        *why = name + " is " + t.dtype + " [" + got + "]; the weight needs " + dtype + " [" +
               want + "]";
        return RAD_E_SHAPE;
    }
    return s->read(t, dst, why) ? RAD_OK : RAD_E_IO;
}

/* ============================================================ exl3i: EXL3-V3's weights (m / 16)
 *
 *   blk.*.ffn_down.weight    exl3i src=$LEVEL_DIR codebook=mad
 *
 * IMPORTS a trellis written by the EXL3-V3 quantizer (rad-exl3-mix quantize/ourq.py: exllamav3's
 * layout, one level a directory, fractional K included) or by exllamav3 into the exl3i encoding
 * (include/exl3i_encoding.h). Nothing is quantised. The rate comes from the trellis' last
 * dimension, 16 K = m (even, 2..128; a stacked weight's parts must agree; rate= checks it when
 * given).
 *
 * The codebook cannot be read off the trellis -- levels of any codebook leave the same trellis --
 * so codebook= is required (mad: the one codebook) and checked against the level's stamp.json
 * when the directory has one (ourq's settings):
 *   mad  = mad (rxq / ourq --codebook mad); the a / b the stamp records (args.mad or
 *          args.mad_a / mad_b) must be exl3i's;
 *   hash = mad (the stage-3 levels: ourq / rxq --codebook hash, the default A and B with no
 *          output map); ourq's args.hash tag, when there, must say so;
 *   any other value has no exl3i codebook here and is refused.
 * A level's `<name>.mul1` file (the exllamav3 layout's marker, written by the exporter) is not
 * read: codebook= and the stamp are authoritative.
 *
 * The weight names are those listed above, plus token_embd.weight (the source's
 * model.language_model.embed_tokens trellis, e.g. ourq's head path run on the token table with an
 * identity input Hessian and its keys renamed): served by exl3i's embed_exl3i, which decodes the
 * token's 128-row block from the stored record (no load-time expansion). */

struct OptsI {
    std::string src;
    int m = -1;    /* rate=: m / 16, or -1 (the source's) */
    int cb = -1;   /* codebook=, required */
    int r16 = 0;   /* rows=r16: rows stored in R16 order (the head; exl3i_encoding.h, /r16) */
};

const char kNameI[] = "exl3i";
const RadQuantOption kOptionsI[] = {
    {"src", RAD_P_STR, nullptr,
     "the level directory the planes are imported from: ourq.py's output (or any exllamav3 "
     "layout): config.json, model.safetensors.index.json and safetensors shards"},
    {"codebook", RAD_P_STR, nullptr,
     "mad (required: the trellis does not say which; checked against src/stamp.json when "
     "present)"},
    {"rate", RAD_P_STR, nullptr,
     "K as text (2.25, 0.5, 4): every imported trellis must be at this rate"},
    {"rows", RAD_P_STR, nullptr,
     "natural (default) | r16: the rows were quantized in R16 order (ourq --head-perm R16.npy, "
     "the head; checked against src/stamp.json and the permutation file when readable) and "
     "stay so; the kernel un-permutes its output"},
};

bool read_opts_i(const RadParam* o, int n, OptsI* out, std::string* why) {
    for (int i = 0; i < n; ++i) {
        const char* k = o[i].key;
        const bool str = o[i].kind == RAD_P_STR && o[i].sval;
        if (!std::strcmp(k, "src") && str) out->src = o[i].sval;
        else if (!std::strcmp(k, "codebook") && str) {
            out->cb = exl3i_codebook_from_name(o[i].sval);
            if (out->cb < 0) {
                *why = std::string("codebook=") + o[i].sval + " is not mad (the one codebook)";
                return false;
            }
        } else if (!std::strcmp(k, "rate") && str) {
            out->m = exl3i_rate_parse(o[i].sval);
            if (out->m < 0) {
                *why = std::string("rate=") + o[i].sval +
                       " is not a rate (1/8 steps, 0.125 .. 8)";
                return false;
            }
        } else if (!std::strcmp(k, "rows") && str) {
            if (!std::strcmp(o[i].sval, "r16")) out->r16 = 1;
            else if (!std::strcmp(o[i].sval, "natural")) out->r16 = 0;
            else {
                *why = std::string("rows=") + o[i].sval + " is not natural or r16";
                return false;
            }
        } else {
            *why = std::string("option '") + k +
                   "' is not one this quantiser takes (src, codebook, rate, rows)";
            return false;
        }
    }
    if (out->src.empty()) { *why = "src=DIR (the level) is required"; return false; }
    if (out->cb < 0) {
        *why = "codebook= is required (mad: the trellis does not say)";
        return false;
    }
    return true;
}

/* The last run's args of the level's stamp.json (ourq.py: a list of runs). 1 with `j` holding the
 * file and *args that object (null when the run has none); 0 when the directory has no stamp
 * (exllamav3 / turboderp levels); -1 with `why` when the file is not a stamp list. */
int read_stamp(const std::string& dir, exl3::Json* j, const exl3::Json** args, std::string* why) {
    std::string text, err;
    if (!exl3::read_file(dir + "/stamp.json", &text, &err)) return 0;
    if (!exl3::parse_json(text.data(), text.data() + text.size(), j, &err) ||
        j->kind != exl3::Json::Arr || j->arr.empty()) {
        *why = dir + "/stamp.json: not ourq's stamp list" + (err.empty() ? "" : " (" + err + ")");
        return -1;
    }
    *args = j->arr.back().get("args");
    return 1;
}

/* A stamp number: a JSON number, or a string holding one (any base strtoull reads). */
bool stamp_num(const exl3::Json* j, unsigned long long* v) {
    if (!j) return false;
    if (j->kind == exl3::Json::Num) { *v = (unsigned long long)j->num; return true; }
    if (j->kind != exl3::Json::Str) return false;
    char* e = nullptr;
    *v = std::strtoull(j->str.c_str(), &e, 0);
    return e && !*e && !j->str.empty();
}

/* The level's own record of its codebook (stamp.json, the last run's args.codebook): true when it
 * agrees with cb or the directory has no stamp; false with `why` otherwise. */
bool stamp_agrees(const std::string& dir, int cb, std::string* why) {
    exl3::Json j;
    const exl3::Json* args = nullptr;
    const int st = read_stamp(dir, &j, &args, why);
    if (st <= 0) return st == 0;
    const exl3::Json* c = args ? args->get("codebook") : nullptr;
    const std::string have = c && c->kind == exl3::Json::Str ? c->str : std::string();
    if (have.empty()) {
        *why = dir + "/stamp.json: the last run records no codebook (the level's codebook is not "
                     "readable)";
        return false;
    }
    const int hc = have == "mad" || have == "hash" ? EXL3I_CB_MAD : -1;   /* else: none here */
    if (hc == EXL3I_CB_MAD && args) {
        /* The code's constants where the level records them -- rxq: args.mad {a, b} as hex text;
         * ourq --codebook mad: args.mad_a / mad_b; ourq --codebook hash: args.hash "A-B[-map]" --
         * must be exl3i's (no output map). */
        char want[32];
        std::snprintf(want, sizeof want, "%06x-%08x", (unsigned)EXL3I_MAD_A,
                      (unsigned)EXL3I_MAD_B);
        const exl3::Json* t = args->get("hash");
        const exl3::Json* m = args->get("mad");
        unsigned long long a = EXL3I_MAD_A, b = EXL3I_MAD_B;
        bool ok = true;
        if (t) ok = t->kind == exl3::Json::Str && t->str == want;
        if (ok && m) ok = m->kind == exl3::Json::Obj && stamp_num(m->get("a"), &a) &&
                          stamp_num(m->get("b"), &b);
        if (ok && args->get("mad_a")) ok = stamp_num(args->get("mad_a"), &a);
        if (ok && args->get("mad_b")) ok = stamp_num(args->get("mad_b"), &b);
        if (!ok || a != EXL3I_MAD_A || b != EXL3I_MAD_B) {
            *why = dir + " was quantized with another mad / hash code (stamp.json's a, b or hash "
                         "tag); codebook=mad is " + want;
            return false;
        }
    }
    if (hc != cb) {
        *why = dir + " was quantized with codebook " + have + " (stamp.json); the recipe says "
               "codebook=" + exl3i_codebook_name(cb);
        return false;
    }
    return true;
}

/* A C-order .npy vector of int64 or int32 values, read as int64. */
bool read_npy_i64(const std::string& path, std::vector<int64_t>* v, std::string* why) {
    std::string t, err;
    if (!exl3::read_file(path, &t, &err)) { *why = path + ": " + err; return false; }
    if (t.size() < 12 || t.compare(0, 6, "\x93NUMPY") != 0) {
        *why = path + ": not a .npy file";
        return false;
    }
    const auto byte = [&t](size_t i) { return (size_t)(unsigned char)t[i]; };
    const bool v1 = byte(6) == 1;
    const size_t hl = v1 ? byte(8) | byte(9) << 8
                         : byte(8) | byte(9) << 8 | byte(10) << 16 | byte(11) << 24;
    const size_t off = (v1 ? 10 : 12) + hl;
    if (off > t.size()) { *why = path + ": a truncated .npy header"; return false; }
    const std::string h = t.substr(0, off);
    const bool i8 = h.find("'<i8'") != std::string::npos;
    const bool i4 = h.find("'<i4'") != std::string::npos;
    if ((!i8 && !i4) || h.find("'fortran_order': False") == std::string::npos) {
        *why = path + ": not a C-order int64 / int32 vector";
        return false;
    }
    const size_t es = i8 ? 8 : 4, n = (t.size() - off) / es;
    v->resize(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t x = 0;
        if (i8) std::memcpy(&x, t.data() + off + i * 8, 8);
        else { int32_t y; std::memcpy(&y, t.data() + off + i * 4, 4); x = y; }
        (*v)[i] = x;
    }
    return true;
}

/* The level's record of a row permutation (ourq.py --head-perm: stamp.json's last run,
 * args.head_perm, the .npy path; stored row j = natural row perm[j]) against rows=: r16 wants one
 * whose file, when readable, is exactly the R16 closed form (exl3i_r16_row) for `rows` rows;
 * natural wants none. True when they agree (or the directory has no stamp and rows=natural). */
bool perm_agrees(const std::string& dir, int r16, int64_t rows, std::string* why) {
    exl3::Json j;
    const exl3::Json* args = nullptr;
    const int st = read_stamp(dir, &j, &args, why);
    if (st < 0) return false;
    if (st == 0) {
        if (r16) {
            *why = dir + " has no stamp.json: rows=r16 wants ourq's record of the head "
                         "permutation";
            return false;
        }
        return true;
    }
    const exl3::Json* hp = args ? args->get("head_perm") : nullptr;
    const bool has = hp && hp->kind == exl3::Json::Str && !hp->str.empty();
    if (!r16) {
        if (has) {
            *why = dir + " was quantized with head_perm " + hp->str + " (stamp.json): give "
                   "rows=r16";
            return false;
        }
        return true;
    }
    if (!has) { *why = "rows=r16, but " + dir + "/stamp.json records no head_perm"; return false; }
    std::vector<int64_t> p;
    std::string e2;
    if (!read_npy_i64(hp->str, &p, &e2)) {
        std::fprintf(stderr, "exl3i: %s (the stamp's head_perm) unreadable here (%s): rows=r16 "
                             "taken from the recipe\n", hp->str.c_str(), e2.c_str());
        return true;
    }
    if ((int64_t)p.size() != rows) {
        *why = hp->str + " has " + std::to_string(p.size()) + " rows; the weight " +
               std::to_string(rows);
        return false;
    }
    for (int64_t s = 0; s < rows; ++s)
        if (p[s] != exl3i_r16_row(s, rows)) {
            *why = hp->str + " is not the R16 order (stored row " + std::to_string(s) +
                   " = natural row " + std::to_string(p[s]) + ", R16 says " +
                   std::to_string(exl3i_r16_row(s, rows)) + ")";
            return false;
        }
    return true;
}

/* m of one projection's trellis (I16 [k/16, n/16, m]). */
int trellis_m(exl3::Source* s, const std::string& base, int64_t k, int64_t n, exl3::Tensor* t,
              std::string* why) {
    const int f = s->find(base + ".trellis", t, why);
    if (f < 0) return RAD_E_IO;
    if (f == 0) { *why = base + ".trellis is not in the source"; return RAD_E_NOTFOUND; }
    if (t->dtype != "I16" || t->shape.size() != 3) {
        *why = base + ".trellis is " + t->dtype + " of rank " + std::to_string(t->shape.size()) +
               "; want I16 [k/16, n/16, 16 K]";
        return RAD_E_FORMAT;
    }
    const int64_t m = t->shape[2];
    if (m > 1000 || !exl3i_rate_ok((int)m)) {
        *why = base + ".trellis has a last dimension of " + std::to_string(m) +
               ": not 16 K for a rate in 1/8 steps up to 8";
        return RAD_E_FORMAT;
    }
    if (t->shape[0] * 16 != k || t->shape[1] * 16 != n) {
        *why = base + ".trellis is [" + std::to_string(t->shape[0]) + ", " +
               std::to_string(t->shape[1]) + ", " + std::to_string(m) + "]: tiles of a [" +
               std::to_string(t->shape[0] * 16) + ", " + std::to_string(t->shape[1] * 16) +
               "] weight; this part is k=" + std::to_string(k) + " in, n=" + std::to_string(n) +
               " out";
        return RAD_E_SHAPE;
    }
    return (int)m;
}

/* The checkpoint name of projection `i` of a weight under `prefix`. */
std::string part_base(const char* prefix, const Name& nm, int i) {
    return prefix[0] ? std::string(prefix) + "." + nm.part[i] : std::string(nm.part[i]);
}

int resolve_i(const RadQuantWeight* w, const Name& nm, const OptsI& op, exl3::Source* s,
              const char* prefix, int* m, std::string* why) {
    const int n_part = nm.part[1] ? 2 : 1;
    *m = -1;
    for (int i = 0; i < n_part; ++i) {
        exl3::Tensor t;
        const int mm = trellis_m(s, part_base(prefix, nm, i), w->cols,
                                 n_part == 2 ? w->rows / 2 : w->rows, &t, why);
        if (mm < 0) return mm;
        if (*m >= 0 && *m != mm) {
            char a[16], b[16];
            *why = "a stacked weight's parts are K=" + std::string(exl3i_rate_str(*m, a)) +
                   " and K=" + exl3i_rate_str(mm, b);
            return RAD_E_DTYPE;
        }
        *m = mm;
    }
    if (op.m > 0 && op.m != *m) {
        char a[16], b[16];
        *why = std::string("is K=") + exl3i_rate_str(*m, a) + " in the source; the recipe says "
               "rate=" + exl3i_rate_str(op.m, b);
        return RAD_E_DTYPE;
    }
    return RAD_OK;
}

/* The shared front of encoding / quantize: name, options, source, stamp, prefix, rate. */
int front_i(const RadParam* o, int n_o, const RadQuantWeight* w, Name* nm, OptsI* op,
            exl3::Source** s, char* prefix, size_t np, int* m) {
    std::string why;
    if (!parse_name(w->name, nm) || nm->kind == Name::kMoE) return RAD_E_UNSUPPORTED;
    if (nm->fused) {
        say(w->name, "ssm_in is served as ssm_qkv beside plain a/b");
        return RAD_E_UNSUPPORTED;
    }
    if (!read_opts_i(o, n_o, op, &why)) { say(w->name, why); return RAD_E_INVAL; }
    if (w->rank != 2 || w->rows <= 0 || w->cols <= 0 || w->rows % EXL3_HAD ||
        w->cols % EXL3_HAD || (nm->stacked && w->rows % (2 * EXL3_HAD))) {
        say(w->name, "a [" + std::to_string(w->rows) + ", " + std::to_string(w->cols) +
                     "] weight: EXL3 needs rows and columns in 128s (both rotations), a stacked "
                     "gate_up's halves too");
        return RAD_E_SHAPE;
    }
    *s = exl3::Source::open(op->src, &why);
    if (!*s) { say(w->name, why); return RAD_E_FORMAT; }
    if (!stamp_agrees(op->src, op->cb, &why)) { say(w->name, why); return RAD_E_DTYPE; }
    if (op->r16 && (nm->stacked || w->rows % 128)) {
        say(w->name, "rows=r16: rows in 128s, not stacked");
        return RAD_E_SHAPE;
    }
    if (!perm_agrees(op->src, op->r16, w->rows, &why)) { say(w->name, why); return RAD_E_DTYPE; }
    int st = prefix_for(w->name, *nm, *s, prefix, np);
    if (st != RAD_OK) return st;
    if ((st = resolve_i(w, *nm, *op, *s, prefix, m, &why)) != RAD_OK) {
        say(w->name, why);
        return st;
    }
    return RAD_OK;
}

int encoding_i(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    Name nm;
    OptsI op;
    exl3::Source* s = nullptr;
    char prefix[160];
    int m = -1;
    const int st = front_i(o, n_o, w, &nm, &op, &s, prefix, sizeof prefix, &m);
    if (st != RAD_OK) return st;
    return op.r16 ? exl3i_encoding_make_r16(m, op.cb, nm.stacked, w->rows, w->cols, out)
                  : exl3i_encoding_make(m, op.cb, nm.stacked, w->rows, w->cols, out);
}

int quantize_i(const RadParam* o, int n_o, const RadQuantWeight* w, float*, int64_t row0,
               int64_t rows, void* const* planes) {
    Name nm;
    OptsI op;
    exl3::Source* s = nullptr;
    char prefix[160];
    int m = -1;
    std::string why;
    if (row0 != 0 || rows != w->rows) {
        say(w->name, "this quantiser takes a weight whole (row_block 0)");
        return RAD_E_STATE;
    }
    const int st = front_i(o, n_o, w, &nm, &op, &s, prefix, sizeof prefix, &m);
    if (st != RAD_OK) return st;
    const int n_part = nm.part[1] ? 2 : 1;
    const int64_t k = w->cols, n = w->rows / n_part, rw = exl3i_row_words(k, m), tw = m / 2;
    uint32_t* rec = static_cast<uint32_t*>(planes[EXL3I_PLANE_RECORD]);
    uint16_t* svh = static_cast<uint16_t*>(planes[EXL3I_PLANE_SVH]);
    for (int i = 0; i < n_part; ++i) {
        const std::string base = part_base(prefix, nm, i);
        exl3::Tensor t;
        if (trellis_m(s, base, k, n, &t, &why) != m) { say(w->name, why); return RAD_E_FORMAT; }
        exl3::Tensor b;
        const int bf = s->find(base + ".bias", &b, &why);
        if (bf != 0) {
            say(w->name, bf < 0 ? why : base + " has a .bias; the EXL3 linears have none");
            return RAD_E_UNSUPPORTED;
        }
        void* suh = planes[i == 0 ? EXL3I_PLANE_SUH : EXL3I_PLANE_SUH1];
        if (fetch(s, base + ".suh", "F16", {k}, suh, &why) != RAD_OK ||
            fetch(s, base + ".svh", "F16", {n}, svh + i * n, &why) != RAD_OK) {
            say(w->name, why);
            return RAD_E_SHAPE;
        }
        /* exllamav3's [k/16][n/16][m] int16 = tile (tk, tn) at (tk nt + tn) m / 2 words ->
         * table row tn, k order */
        std::vector<uint32_t> tr((size_t)(t.bytes() / 4));
        if (!s->read(t, tr.data(), &why)) { say(w->name, why); return RAD_E_IO; }
        const int64_t kt = k / 16, nt = n / 16;
        for (int64_t tn = 0; tn < nt; ++tn)
            for (int64_t tk = 0; tk < kt; ++tk)
                std::memcpy(rec + (i * nt + tn) * rw + tk * tw, tr.data() + (tk * nt + tn) * tw,
                            (size_t)tw * 4);
    }
    return RAD_OK;   /* codes: one zero bit a row, already zeroed */
}

/* exl3i_decode_rows' status (0, -2 shape, -3 memory) as a RadStatus. */
int decode_status(int rc) {
    switch (rc) {
        case 0: return RAD_OK;
        case -2: return RAD_E_SHAPE;
        case -3: return RAD_E_NOMEM;
        default: return RAD_E_INVAL;
    }
}

int decode_i(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes,
             int64_t row0, int64_t rows, float* out) {
    int m = 0, cb = 0, stacked = 0, r16 = 0;
    int64_t r = 0, c = 0;
    const int st = exl3i_encoding_read_r16(e, &m, &cb, &stacked, &r, &c, &r16);
    if (st != RAD_OK) return st;
    if (w && (w->rows != r || w->cols != c)) return RAD_E_SHAPE;
    Exl3iView v;
    std::memset(&v, 0, sizeof v);
    v.rows = r; v.cols = c; v.m = m; v.cb = cb;
    v.record = static_cast<const uint32_t*>(planes[EXL3I_PLANE_RECORD]);
    v.suh = static_cast<const uint16_t*>(planes[EXL3I_PLANE_SUH]);
    v.svh = static_cast<const uint16_t*>(planes[EXL3I_PLANE_SVH]);
    v.suh1 = stacked ? static_cast<const uint16_t*>(planes[EXL3I_PLANE_SUH1]) : nullptr;
    if (!r16) return decode_status(exl3i_decode_rows(&v, row0, rows, out));
    /* Logical (vocabulary) row lv is stored row
     * s = (lv % (r / 8)) / 16 * 128 + lv / (r / 8) * 16 + lv % 16: decode each needed 128-row
     * stored block once (the output rotation spans it) and take its rows. */
    if (row0 < 0 || rows < 0 || row0 + rows > r) return RAD_E_SHAPE;
    std::vector<float> blk((size_t)128 * c);
    int64_t have = -1;
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t lv = row0 + i, e8 = r / 8;
        const int64_t s = (lv % e8) / 16 * 128 + lv / e8 * 16 + lv % 16, b = s / 128;
        if (b != have) {
            const int rc = decode_status(exl3i_decode_rows(&v, b * 128, 128, blk.data()));
            if (rc != RAD_OK) return rc;
            have = b;
        }
        std::memcpy(out + i * c, blk.data() + (s - b * 128) * c, (size_t)c * sizeof(float));
    }
    return RAD_OK;
}

const RadQuantizerInfo kQuant[] = {{
    kNameI,
    "imports EXL3-V3 trellises (any rate in 1/8 steps, mad) from src=DIR (an EXL3-V3 quantizer "
    "level, exllamav3's layout) into the exl3i encoding; decodes for the oracle",
    kOptionsI, (int)(sizeof kOptionsI / sizeof kOptionsI[0]),
    encoding_i, row_block, quantize_i, decode_i,
}};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT, "exl3quant", "0.3.0",
    "the exl3i quantiser: EXL3-V3 levels imported from an exllamav3-layout directory (any rate in "
    "1/8 steps, codebook mad)", "host",
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_quant_count(void) { return 1; }
extern "C" const RadQuantizerInfo* rad_quant_at(int i) { return i == 0 ? &kQuant[0] : nullptr; }
