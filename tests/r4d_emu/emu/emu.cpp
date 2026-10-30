#include "hip/hip_runtime.h"
#include "rad_plugin.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

/* ---- the context switch: callee-saved registers and the stack pointer (x86-64 SysV) */
extern "C" void emu_switch(void** save_sp, void* new_sp);
asm(R"(
.text
.globl emu_switch
.type emu_switch,@function
emu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
)");

namespace {
constexpr size_t kStack = 512 * 1024;
struct Fiber {
    void* sp = nullptr;
    /* left uninitialised so the OS commits only the pages a fiber touches: a 1024-thread block
     * would otherwise zero half a GiB */
    std::unique_ptr<char[]> stack;
    EmuIdx tid{};
    bool done = false;
    unsigned bar_gen = 0;      /* block barrier generation this fiber waits on */
    unsigned wbar_gen = 0;
    int waiting = 0;           /* 0 run, 1 block barrier, 2 wave barrier */
};
struct Sched {
    std::vector<Fiber> f;
    void* main_sp = nullptr;
    int cur = -1, nthreads = 0;
    EmuIdx bid{}, bdim{}, gdim{};
    void (*body)(void*) = nullptr;
    void* ctx = nullptr;
    /* block barrier; or_acc[g % 3] collects __syncthreads_or's predicates at generation g */
    int arrive = 0; unsigned gen = 0;
    int or_acc[3] = {};
    /* wave barriers and exchange slots, double-buffered by parity */
    int warrive[64] = {}; unsigned wgen[64] = {};
    int slot[64][2][32] = {};
    int tag[64][2][32] = {};   /* which op, and its uniform operand, each lane posted */
    unsigned xparity[1024] = {};
} S;

void yield() { Fiber& me = S.f[S.cur]; emu_switch(&me.sp, S.main_sp); }

/* Lanes of wave `w` that have not returned. A cross-lane op on the card runs over the active lanes:
 * a lane that has returned takes no part, and the op completes without it (a read of its value is
 * the kernel's bug, which the exchange poisons below). */
int g_wmma_pending[64];   /* the wave exchange now forming is a WMMA's, which every lane must join */
/* The lanes that had returned when the wave's last exchange completed: their slots hold no value
 * for it. Taken at completion, not at read time -- a lane that posted and then returned before a
 * slower lane read the exchange still contributed. The next completion needs every live lane to
 * arrive, so a lane reads this before it can be overwritten. */
unsigned g_absent[64];
unsigned wave_done_mask(int w);
unsigned wave_done_mask(int w) {
    unsigned m = 0;
    for (int i = 0; i < 32 && w * 32 + i < S.nthreads; ++i) m |= (unsigned)S.f[w * 32 + i].done << i;
    return m;
}
int wave_alive(int w) {
    int n = 0;
    for (int i = w * 32; i < w * 32 + 32 && i < S.nthreads; ++i) n += !S.f[i].done;
    return n;
}
extern "C" void emu_trampoline() {
    S.body(S.ctx);
    S.f[S.cur].done = true;
    /* The lanes still running may be waiting at a wave exchange only this lane had yet to reach. */
    const int w = (int)S.f[S.cur].tid.x >> 5;
    if (S.warrive[w] > 0 && S.warrive[w] == wave_alive(w)) {
        if (g_wmma_pending[w]) {
            std::fprintf(stderr, "emu: block (%u,%u,%u), wave %d: lane %d returned while the others wait at a "
                         "WMMA it skipped (kernel %s)\n", S.bid.x, S.bid.y, S.bid.z, w,
                         (int)S.f[S.cur].tid.x & 31, emu_last_kernel);
            std::abort();
        }
        g_absent[w] = wave_done_mask(w);
        S.warrive[w] = 0; ++S.wgen[w];
    }
    yield();
    std::abort();
}
/* the entry a fresh stack "returns" into, keeping the SysV alignment at call boundaries */
asm(R"(
.text
.globl emu_entry
.type emu_entry,@function
emu_entry:
    andq $-16, %rsp
    call emu_trampoline
    ud2
)");
extern "C" void emu_entry();
}  // namespace

EmuIdx& emu_thread_idx() { return S.f[S.cur].tid; }
EmuIdx& emu_block_idx() { return S.bid; }
EmuIdx& emu_block_dim() { return S.bdim; }
EmuIdx& emu_grid_dim() { return S.gdim; }

void __syncthreads() {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.gen;
    /* Clearing the next generation's slot here, as the barrier completes, keeps it ahead of every
     * write to it (threads arrive at g + 1 only after g completes) and behind every read of the
     * slot it shares a residue with (g - 2's, read before arriving at g - 1). */
    if (++S.arrive == S.nthreads) { S.arrive = 0; S.or_acc[(g + 1) % 3] = 0; ++S.gen; return; }
    me.waiting = 1;
    while (S.gen == g) yield();
    me.waiting = 0;
}
int __syncthreads_or(int p) {
    const unsigned g = S.gen;
    if (p) S.or_acc[g % 3] = 1;
    __syncthreads();
    return S.or_acc[g % 3];
}

static void wave_barrier(int w) {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.wgen[w];
    if (++S.warrive[w] == wave_alive(w)) {
        g_absent[w] = wave_done_mask(w);
        S.warrive[w] = 0; ++S.wgen[w]; g_wmma_pending[w] = 0;
        return;
    }
    me.waiting = 2;
    while (S.wgen[w] == g) yield();
    me.waiting = 0;
}

/* Every lane posts `v` to the wave's exchange slots and passes the wave barrier; the slots of this
 * exchange are returned for the caller to read. `tag` names the op and its wave-uniform operand:
 * lanes that met at different ops (a cross-lane op called under divergence, each side reaching
 * one) are reported, not combined. */
enum { kOpBpermute = 1 << 24, kOpReadlane = 2 << 24, kOpBallot = 3 << 24, kOpDpp = 4 << 24,
       kOpPermlanex16 = 5 << 24 };
static const int* exchange(int v, int* lane, int tag) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5;
    const unsigned par = S.xparity[t]++ & 1u;
    *lane = t & 31;
    S.slot[w][par][*lane] = v;
    S.tag[w][par][*lane] = tag;
    wave_barrier(w);
    for (int i = 0; i < 32; ++i) {
        if ((g_absent[w] >> i) & 1u) {       /* a lane that had returned: no value; a read of it shows */
            S.slot[w][par][i] = 0x7fbadbad;
            continue;
        }
        if (S.tag[w][par][i] != tag) {
            std::fprintf(stderr, "emu: DIVERGENCE in block (%u,%u,%u), wave %d: lanes %d and %d met at different "
                         "cross-lane ops (0x%x, 0x%x)\n", S.bid.x, S.bid.y, S.bid.z, w, *lane, i, tag,
                         S.tag[w][par][i]);
            std::abort();
        }
    }
    return S.slot[w][par];
}
int emu_bpermute(int addr, int v) {
    int l;
    return exchange(v, &l, kOpBpermute)[(addr >> 2) & 31];
}
/* v_readlane_b32 takes a wave-uniform lane, so a lane-varying one is reported */
int emu_readlane(int v, int lane) {
    int l;
    return exchange(v, &l, kOpReadlane | (lane & 31))[lane & 31];
}
int emu_cvt_pk_fp8(float a, float b, int old, bool hi) {
    const unsigned pk = (unsigned)rad_f32_to_fp8e4m3(a) | ((unsigned)rad_f32_to_fp8e4m3(b) << 8);
    const unsigned o = (unsigned)old;
    return (int)(hi ? ((o & 0xffffu) | (pk << 16)) : ((o & 0xffff0000u) | pk));
}
/* A WMMA reads every lane's fragment: one issued with lanes already returned is reported. */
static void wmma_whole_wave(int w) {
    g_wmma_pending[w] = 1;
    if (wave_alive(w) != 32) {
        std::fprintf(stderr, "emu: WMMA in block (%u,%u,%u), wave %d with %d of 32 lanes active (kernel %s); returned:",
                     S.bid.x, S.bid.y, S.bid.z, w, wave_alive(w), emu_last_kernel);
        for (int i = w * 32; i < w * 32 + 32; ++i) if (S.f[i].done) std::fprintf(stderr, " lane %d", i & 31);
        std::fprintf(stderr, "\n");
        std::abort();
    }
}
/* One 16x16x16 WMMA: each lane posts its A and B elements, then reads the whole tile back. `ga`
 * and `gb` are the exchange slots of one element type, [wave][parity][lane][element]. */
template <class T, class V, class Cvt>
static emu_v8f emu_wmma(T (&ga)[64][2][32][8], T (&gb)[64][2][32][8], V a, V b, emu_v8f c,
                        Cvt cvt) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    for (int e = 0; e < 8; ++e) { ga[w][par][l][e] = a[e]; gb[w][par][l][e] = b[e]; }
    wmma_whole_wave(w);
    wave_barrier(w);
    const int N = l & 15, h = l >> 4;
    emu_v8f d;
    for (int e = 0; e < 8; ++e) {
        const int M = 8 * h + e;
        float acc = c[e];
        for (int k = 0; k < 16; ++k)
            acc += cvt(ga[w][par][M + 16 * (k >> 3)][k & 7]) * cvt(gb[w][par][N + 16 * (k >> 3)][k & 7]);
        d[e] = acc;
    }
    return d;
}
static short g_wa[64][2][32][8], g_wb[64][2][32][8];
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c) {
    return emu_wmma(g_wa, g_wb, a, b, c, [](short v) { return rad_bf16_to_f32((uint16_t)v); });
}
static _Float16 g_ha[64][2][32][8], g_hb[64][2][32][8];
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c) {
    return emu_wmma(g_ha, g_hb, a, b, c, [](_Float16 v) { return (float)v; });
}
namespace {
struct Bytes8 {
    uint8_t v[8];
    uint8_t operator[](int e) const { return v[e]; }
};
}
static uint8_t g_qa[64][2][32][8], g_qb[64][2][32][8];
emu_v8f emu_wmma_fp8(emu_v2i a, emu_v2i b, emu_v8f c) {
    Bytes8 pa, pb;
    std::memcpy(pa.v, &a, 8);
    std::memcpy(pb.v, &b, 8);
    return emu_wmma(g_qa, g_qb, pa, pb, c, [](uint8_t v) { return rad_fp8e4m3_to_f32(v); });
}
emu_v8i emu_wmma_iu8(bool, emu_v2i, bool, emu_v2i, emu_v8i, bool) {
    std::fprintf(stderr, "emu: v_wmma_i32_16x16x16_iu8 is not emulated\n");
    std::abort();
}
unsigned emu_ballot(bool p) {
    int l;
    const int* s = exchange(p ? 1 : 0, &l, kOpBallot);
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) m |= (unsigned)(s[i] & 1) << i;
    return m;
}
int emu_update_dpp(int old, int src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl) {
    int l;
    const int* s = exchange(src, &l, kOpDpp | (ctrl & 0xffff));
    const int r = l & 15, k = ctrl & 15;
    int from;
    if (ctrl > 0x100 && ctrl < 0x110)      from = r + k < 16 ? l + k : -1;     /* row_shl */
    else if (ctrl > 0x110 && ctrl < 0x120) from = r - k >= 0 ? l - k : -1;     /* row_shr */
    else if (ctrl >= 0x160 && ctrl < 0x170) from = (l & 16) | (r ^ k);         /* row_xmask */
    else { std::fprintf(stderr, "emu: dpp_ctrl 0x%x is not emulated\n", ctrl); std::abort(); }
    if ((row_mask & 3) != 3 || (bank_mask & 15) != 15) {
        std::fprintf(stderr, "emu: dpp row/bank masks 0x%x/0x%x are not emulated\n", row_mask, bank_mask);
        std::abort();
    }
    return from < 0 ? (bound_ctrl ? 0 : old) : s[from];
}
int emu_permlanex16(int, int src, unsigned sel_lo, unsigned sel_hi, bool, bool) {
    int l;
    const int* s = exchange(src, &l, kOpPermlanex16);
    const int r = l & 15;
    const unsigned sel = (r < 8 ? sel_lo >> (4 * r) : sel_hi >> (4 * (r - 8))) & 15u;
    return s[(16 - (l & 16)) | (int)sel];
}
emu_v8s emu_load_tr_b128(const void*) {
    std::fprintf(stderr, "emu: global_load_tr_b128 is not emulated\n");
    std::abort();
}

/* Dynamic LDS: a mapping whose last page is inaccessible, the buffer placed against it with its
 * start 16-byte aligned, refilled with a poison pattern for every block. */
static char* g_lds_map = nullptr;
static size_t g_lds_map_len = 0;
static char* g_lds = nullptr;
void* emu_dyn_lds() { return g_lds; }
static void lds_reserve(size_t bytes) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t len = (bytes + 15 + page - 1) / page * page + page;
    if (len > g_lds_map_len) {
        if (g_lds_map) munmap(g_lds_map, g_lds_map_len);
        void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED || mprotect((char*)m + len - page, page, PROT_NONE) != 0) {
            std::fprintf(stderr, "emu: no dynamic LDS mapping\n");
            std::abort();
        }
        g_lds_map = (char*)m;
        g_lds_map_len = len;
    }
    const uintptr_t end = (uintptr_t)(g_lds_map + g_lds_map_len - page);
    g_lds = (char*)((end - bytes) & ~(uintptr_t)15);
}

int emu_grids = 0;
int emu_last_block = 0;
const char* emu_last_kernel = "";
void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx, size_t dyn_lds) {
    ++emu_grids;
    const int n = (int)(block.x * block.y * block.z);
    emu_last_block = n;
    if (block.y != 1 || block.z != 1 || n % 32 || n > 1024) { std::fprintf(stderr, "emu: block shape\n"); std::abort(); }
    if (dyn_lds > 64 * 1024) {
        std::fprintf(stderr, "emu: %zu bytes of dynamic LDS, past the 64 KiB a workgroup has\n", dyn_lds);
        std::abort();
    }
    lds_reserve(dyn_lds);
    S.body = body; S.ctx = ctx; S.nthreads = n;
    S.bdim = EmuIdx{block.x, block.y, block.z};
    S.gdim = EmuIdx{grid.x, grid.y, grid.z};
    S.f.resize(n);
    for (auto& f : S.f) if (!f.stack) f.stack.reset(new char[kStack]);
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        S.bid = EmuIdx{bx, by, bz};
        std::memset(g_lds, 0xA5, dyn_lds);
        S.arrive = 0; S.gen = 0;
        for (int& a : S.or_acc) a = 0;
        for (int w = 0; w < 64; ++w) { S.warrive[w] = 0; S.wgen[w] = 0; g_wmma_pending[w] = 0; g_absent[w] = 0; }
        for (int t = 0; t < n; ++t) {
            Fiber& f = S.f[t];
            f.done = false; f.waiting = 0; f.tid = EmuIdx{(unsigned)t, 0, 0};
            S.xparity[t] = 0;
            /* a fresh stack: six callee-saved slots, then the entry as the return address */
            void** top = (void**)(f.stack.get() + kStack - 64);
            *--top = (void*)&emu_entry;
            for (int r = 0; r < 6; ++r) *--top = nullptr;
            f.sp = top;
        }
        int live = n;
        while (live > 0) {
            const int live_before = live;
            const unsigned g0 = S.gen; unsigned wg0[64]; for (int w = 0; w < 64; ++w) wg0[w] = S.wgen[w];
            int arr0 = S.arrive; int warr0[64]; for (int w = 0; w < 64; ++w) warr0[w] = S.warrive[w];
            live = 0;
            for (int t = 0; t < n; ++t) {
                if (S.f[t].done) continue;
                S.cur = t;
                emu_switch(&S.main_sp, S.f[t].sp);
                S.cur = -1;
                if (!S.f[t].done) ++live;
            }
            bool progressed = S.gen != g0 || S.arrive != arr0 || live != live_before;
            for (int w = 0; w < 64; ++w) progressed |= (S.wgen[w] != wg0[w]) || (S.warrive[w] != warr0[w]);
            if (live > 0 && !progressed) {
                int done = 0, wb = 0, bb = 0;
                for (auto& f : S.f) { done += f.done; wb += f.waiting == 2; bb += f.waiting == 1; }
                std::fprintf(stderr, "emu: DEADLOCK in block (%u,%u,%u): %d threads done, %d at a block "
                             "barrier, %d at a wave barrier -- a barrier not reached uniformly (kernel %s)\n",
                             bx, by, bz, done, bb, wb, emu_last_kernel);
                std::abort();
            }
        }
    }
}

/* ---- gfx11 (gfx1151) WMMA forms, for libhalo (halo_wmma.h's layout) -------------------------- */
/* Each half-wave computes its own rows from its own lanes, as measured on the part: D's row M lives
 * in half h = M & 1 and takes A's row M from lane M + 16h and B from that half's lanes. A kernel
 * that replicates its operands across the halves (the documented form) sees no difference; one that
 * leaves the lanes the part never reads holding something else is emulated as the part runs it. */
static uint16_t g11_a[64][2][32][16], g11_b[64][2][32][16];
template <class Cvt>
static emu_v8f emu_wmma11(const uint16_t (&a)[16], const uint16_t (&b)[16], emu_v8f c, Cvt cvt) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    for (int e = 0; e < 16; ++e) { g11_a[w][par][l][e] = a[e]; g11_b[w][par][l][e] = b[e]; }
    wmma_whole_wave(w);
    wave_barrier(w);
    const int N = l & 15, h = l >> 4;
    emu_v8f d;
    for (int e = 0; e < 8; ++e) {
        const int M = 2 * e + h;
        float acc = c[e];
        for (int k = 0; k < 16; ++k)
            acc += cvt(g11_a[w][par][M + 16 * h][k]) * cvt(g11_b[w][par][N + 16 * h][k]);
        d[e] = acc;
    }
    return d;
}
emu_v8f emu_wmma11_bf16(emu_v16bf a, emu_v16bf b, emu_v8f c) {
    uint16_t ua[16], ub[16];
    std::memcpy(ua, &a, 32);
    std::memcpy(ub, &b, 32);
    return emu_wmma11(ua, ub, c, [](uint16_t v) { return rad_bf16_to_f32(v); });
}
emu_v8f emu_wmma11_f16(emu_v16h a, emu_v16h b, emu_v8f c) {
    uint16_t ua[16], ub[16];
    std::memcpy(ua, &a, 32);
    std::memcpy(ub, &b, 32);
    return emu_wmma11(ua, ub, c, [](uint16_t v) { _Float16 h; std::memcpy(&h, &v, 2); return (float)h; });
}
emu_v2hf emu_cvt_pkrtz(float a, float b) {
    auto rtz = [](float f) {
        _Float16 h = (_Float16)f;                       /* nearest; step back toward zero if it rounded away */
        if (std::fabs((float)h) > std::fabs(f)) {
            uint16_t u; std::memcpy(&u, &h, 2); --u; std::memcpy(&h, &u, 2);
        }
        __fp16 r; std::memcpy(&r, &h, 2); return r;
    };
    emu_v2hf v; v[0] = rtz(a); v[1] = rtz(b); return v;
}
static uint8_t g11_qa[64][2][32][16], g11_qb[64][2][32][16];
emu_v8i emu_wmma11_iu8(bool sa, emu_v4i a, bool sb, emu_v4i b, emu_v8i c, bool) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    std::memcpy(g11_qa[w][par][l], &a, 16);
    std::memcpy(g11_qb[w][par][l], &b, 16);
    wmma_whole_wave(w);
    wave_barrier(w);
    const int N = l & 15, h = l >> 4;
    emu_v8i d;
    for (int e = 0; e < 8; ++e) {
        const int M = 2 * e + h;
        int acc = c[e];
        for (int k = 0; k < 16; ++k) {
            const int x = sa ? (int)(int8_t)g11_qa[w][par][M][k] : (int)g11_qa[w][par][M][k];
            const int y = sb ? (int)(int8_t)g11_qb[w][par][N][k] : (int)g11_qb[w][par][N][k];
            acc += x * y;
        }
        d[e] = acc;
    }
    return d;
}
emu_v2f emu_cvt_pk_f32_fp8(int w, bool hi) {
    const unsigned u = (unsigned)w >> (hi ? 16 : 0);
    emu_v2f r;
    r[0] = rad_fp8e4m3_to_f32((uint8_t)(u & 0xffu));
    r[1] = rad_fp8e4m3_to_f32((uint8_t)((u >> 8) & 0xffu));
    return r;
}
