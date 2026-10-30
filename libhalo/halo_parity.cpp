// halo_parity <libr4d.so> <libhalo.so> -- is libhalo the same plugin interface as libr4d?
//
// libhalo's promise is that every model plugin runs on it unchanged: the same op schemas, the
// same rows (name, op, domain, priority, constraints, tunables, and the declaration order that
// breaks a priority tie), so a recipe, a tuning cache or a selection that names a libr4d row finds
// the same row here, chosen the same way. Kernel bodies, scratch sizes and launch
// promises are each library's own, measured on its card. This dlopens both plugins (host code
// only: the tables and the promise symbols, no device calls) and fails on any difference in the
// interface; differences in the optional hooks and in the chain promise are printed, not failed,
// because those are per-library choices (a row may size a scratch libr4d's does not need).
//
// Built only when one build produces both plugins (RAD_GPU_TARGETS naming a gfx12 and gfx1151).
#include "rad_abi.h"

#include <dlfcn.h>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

struct Plugin {
    void* h = nullptr;
    int (*kc)(void) = nullptr;
    const RadKernelInfo* (*ka)(int) = nullptr;
    int (*sc)(void) = nullptr;
    const RadOpSchema* (*sa)(int) = nullptr;
    int (*conc)(int) = nullptr;
    int (*chain)(int) = nullptr;
    int (*coll)(int) = nullptr;

    bool open(const char* path) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "halo_parity: %s\n", dlerror()); return false; }
        kc = (int (*)(void))dlsym(h, "rad_kernel_count");
        ka = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        sc = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
        sa = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        conc = (int (*)(int))dlsym(h, "rad_kernel_concurrent");
        chain = (int (*)(int))dlsym(h, "rad_kernel_chain");
        coll = (int (*)(int))dlsym(h, "rad_kernel_collective");
        if (!kc || !ka) { std::fprintf(stderr, "halo_parity: %s has no row table\n", path); return false; }
        return true;
    }
};

std::string schema_str(const RadOpSchema* s) {
    std::string l = "params=";
    for (int j = 0; j < s->n_params; ++j)
        l += std::string(s->params[j].key) + ":" + std::to_string(s->params[j].type) + ":" +
             std::to_string(s->params[j].required) + ":" + std::to_string(s->params[j].role) + ",";
    l += " operands=";
    for (int j = 0; j < s->n_operands; ++j)
        l += std::string(s->operands[j].name) + ":" + std::to_string(s->operands[j].role) + ":" +
             std::to_string(s->operands[j].optional) + ",";
    return l;
}

/* The interface of a row: what selection, the tuner and a recipe can see. Constraints are kept
 * in declaration order; a reorder is a difference too (it changes which one a miss reports). */
std::string row_str(const RadKernelInfo* k) {
    static const char* ops[] = { "EQ", "LE", "GE", "DIV", "IN" };
    std::string l = std::string("op=") + k->op + " dom=" + std::to_string(k->domain) +
                    " prio=" + std::to_string(k->priority) + " C={";
    for (int j = 0; j < k->n_constraints; ++j) {
        const RadConstraint& c = k->constraints[j];
        l += std::string(c.key) + " " + (c.op >= 0 && c.op <= 4 ? ops[c.op] : "?") + " " +
             (c.op == RAD_C_IN ? std::string(c.sval ? c.sval : "") : std::to_string(c.ival)) + "; ";
    }
    l += "} T={";
    for (int j = 0; j < k->n_tunables; ++j) {
        const RadTunable& t = k->tunables[j];
        l += std::string(t.key) + "[";
        for (int q = 0; q < t.n_values; ++q)
            l += std::to_string((long long)t.values[q]) + (q + 1 < t.n_values ? "," : "");
        l += "]=" + std::to_string((long long)t.deflt) + " ";
    }
    return l + "}";
}

std::string hooks_str(const RadKernelInfo* k) {
    std::string l;
    l += k->init ? 'I' : '-';
    l += k->scratch ? 'S' : '-';
    l += k->layout ? 'L' : '-';
    l += k->relayout ? 'R' : '-';
    l += k->opd_shape ? 'O' : '-';
    l += k->unrelayout ? 'U' : '-';
    l += k->replaces ? 'F' : '-';
    l += k->tune_valid ? 'V' : '-';
    return l;
}

struct Rows {
    std::map<std::string, std::string> schema, row, hooks, conc, coll, chain;
    std::map<std::string, std::string> order;   /* op -> its rows in declaration order */
};

Rows read(const Plugin& p) {
    Rows r;
    if (p.sc && p.sa)
        for (int i = 0; i < p.sc(); ++i) r.schema[p.sa(i)->op] = schema_str(p.sa(i));
    for (int i = 0; i < p.kc(); ++i) {
        const RadKernelInfo* k = p.ka(i);
        r.row[k->name] = row_str(k);
        r.hooks[k->name] = hooks_str(k);
        r.conc[k->name] = std::to_string(p.conc ? p.conc(i) : 0);
        r.coll[k->name] = std::to_string(p.coll ? p.coll(i) : 0);
        r.chain[k->name] = std::to_string(p.chain ? p.chain(i) : 0);
        r.order[k->op] += std::string(k->name) + " ";
    }
    return r;
}

/* Prints every key whose value differs or that one side lacks; returns the count. */
int compare(const char* what, const std::map<std::string, std::string>& a,
            const std::map<std::string, std::string>& b, bool fail) {
    int n = 0;
    for (const auto& [k, v] : a) {
        auto it = b.find(k);
        if (it == b.end()) { std::printf("%s %s %s: only in libr4d\n", fail ? "FAIL" : "info", what, k.c_str()); ++n; }
        else if (it->second != v) {
            std::printf("%s %s %s:\n  libr4d  %s\n  libhalo %s\n", fail ? "FAIL" : "info", what,
                        k.c_str(), v.c_str(), it->second.c_str());
            ++n;
        }
    }
    for (const auto& [k, v] : b)
        if (!a.count(k)) { std::printf("%s %s %s: only in libhalo\n", fail ? "FAIL" : "info", what, k.c_str()); ++n; }
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: halo_parity <libr4d.so> <libhalo.so>\n"); return 2; }
    Plugin r4d, halo;
    if (!r4d.open(argv[1]) || !halo.open(argv[2])) return 1;
    const Rows a = read(r4d), b = read(halo);
    int fails = 0;
    fails += compare("schema", a.schema, b.schema, true);
    fails += compare("row", a.row, b.row, true);
    fails += compare("concurrent", a.conc, b.conc, true);
    fails += compare("collective", a.coll, b.coll, true);
    fails += compare("row order", a.order, b.order, true);
    const int hooks = compare("hooks", a.hooks, b.hooks, false);
    const int chain = compare("chain", a.chain, b.chain, false);
    std::printf("halo_parity: %zu schemas, %zu rows; %d interface difference(s); %d hook and %d chain "
                "difference(s) (per-library, not failed)\n",
                b.schema.size(), b.row.size(), fails, hooks, chain);
    return fails ? 1 : 0;
}
