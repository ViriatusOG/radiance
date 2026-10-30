/* tests/halo_emu (from tests/r4d_emu/test_escale.cpp): expert_scale (halo_expert_scale_bf16.hip) on the host SIMT emulator, checked bit
 * for bit against the chain it replaces (embed_lookup of the [n_expert, 1] scale table by the
 * ids, then mul into the weights) and against ref_host.cpp; one product rounded once has a single
 * correct result. Ids outside the table (a negative pad, one past the end) read a zero scale;
 * rows past M and the pitch's padding stay untouched. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0, cases = 0;
    for (int ne : {32, 128}) for (int k : {1, 8}) for (int M : {1, 4, 16, 37}) for (int pad : {0, 3}) {
        const int ld = k + pad, R = M + 2;
        std::vector<int32_t> ids((size_t)R * ld);
        for (auto& v : ids) {
            const int u = (int)(rng() % 40);
            v = u == 0 ? -1 : u == 1 ? ne : (int)(rng() % (unsigned)ne);      /* pads, one past */
        }
        std::vector<uint16_t> table((size_t)ne), ew0((size_t)R * ld);
        for (auto& v : table) v = f2b(0.5f + 0.25f * nd(rng));
        for (auto& v : ew0) v = f2b(nd(rng));
        auto dev = ew0, hst = ew0;
        RadTensor t[3]{};
        t[0].data = ids.data(); t[0].dtype = RAD_I32; t[0].rank = 2; t[0].shape[0] = M; t[0].shape[1] = k; t[0].stride[0] = ld; t[0].stride[1] = 1;
        t[1].data = table.data(); t[1].dtype = RAD_BF16; t[1].rank = 1; t[1].shape[0] = ne; t[1].stride[0] = 1;
        t[2].dtype = RAD_BF16; t[2].rank = 2; t[2].shape[0] = M; t[2].shape[1] = k; t[2].stride[0] = ld; t[2].stride[1] = 1;
        RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n_expert", RAD_P_INT, ne}; p[2] = {"top_k", RAD_P_INT, k};
        p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
        t[2].data = dev.data();
        const int rd = halo_rad_expert_scale_bf16(&a, nullptr);
        t[2].data = hst.data();
        const int rh = host_expert_scale(&a);
        bool ok = rd == RAD_OK && rh == RAD_OK;
        for (int r = 0; r < R && ok; ++r)
            for (int c = 0; c < ld && ok; ++c) {
                const size_t o = (size_t)r * ld + c;
                uint16_t want = ew0[o];
                if (r < M && c < k) {
                    const int id = ids[o];
                    const uint16_t es = (id >= 0 && id < ne) ? table[(size_t)id] : f2b(0.0f);   /* embed_lookup */
                    want = f2b(b2f(ew0[o]) * b2f(es));                                           /* mul */
                }
                ok = dev[o] == want && hst[o] == want;
            }
        ++cases;
        if (!ok) { ++fails; std::printf("FAIL n_expert %d top_k %d M %d pitch %d\n", ne, k, M, ld); }
    }
    std::printf("expert_scale: %d cases, %d failure(s)\n", cases, fails);
    return fails != 0;
}
