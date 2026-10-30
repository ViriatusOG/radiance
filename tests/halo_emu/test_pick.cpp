/* pair_argmax: device kernel (emulated) vs ref_host.cpp vs brute force, over the row-interleaved
 * [rows][world][R][2] layout the all_gather produces -- ties, negative (unfilled) ids, -inf, empty rows. */
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
    std::mt19937 rng(11);
    int fails = 0, cases = 0;
    for (int world : {1, 2, 4, 8}) for (int R : {2, 4}) for (int rows : {1, 7, 16, 33, 240}) for (int rank3 = 0; rank3 < 2; ++rank3) {
        const int n = world * R;
        std::vector<float> pairs((size_t)rows * n * 2);
        std::vector<int32_t> ref(rows), dev(rows, 12345), hst(rows, 12345), pitched(rows * 3, 777);
        std::vector<float> dval(rows), hval(rows);
        for (int r = 0; r < rows; ++r) {
            int32_t bid = -1; float bv = -INFINITY;
            const int mode = rng() % 5;   /* 0: ties, 1: some unfilled, 2: all unfilled, else random */
            for (int j = 0; j < n; ++j) {
                int32_t id = (int32_t)(rng() % 262144);
                float v = mode == 0 ? 3.5f : std::ldexp((float)(int)(rng() % 2000) - 1000.f, -4);
                if ((mode == 1 && rng() % 3 == 0) || mode == 2) { id = -1; v = -INFINITY; }
                std::memcpy(&pairs[((size_t)r * n + j) * 2], &id, 4);
                pairs[((size_t)r * n + j) * 2 + 1] = v;
                if (id >= 0 && (bid < 0 || v > bv || (v == bv && id < bid))) { bid = id; bv = v; }
            }
            ref[r] = bid;
        }
        RadTensor t[3]{};
        t[0].data = pairs.data(); t[0].dtype = RAD_F32;
        if (rank3) { t[0].rank = 3; t[0].shape[0] = rows; t[0].shape[1] = n; t[0].shape[2] = 2; t[0].stride[0] = n * 2; t[0].stride[1] = 2; t[0].stride[2] = 1; }
        else       { t[0].rank = 2; t[0].shape[0] = rows; t[0].shape[1] = 2 * n; t[0].stride[0] = 2 * n; t[0].stride[1] = 1; }
        t[1].dtype = RAD_I32; t[1].rank = 2; t[1].shape[0] = rows; t[1].shape[1] = 1; t[1].stride[0] = 1; t[1].stride[1] = 1;
        t[2].dtype = RAD_F32; t[2].rank = 2; t[2].shape[0] = rows; t[2].shape[1] = 1; t[2].stride[0] = 1; t[2].stride[1] = 1;
        RadParam p[3]{}; p[0] = {"M", RAD_P_INT, rows}; p[1] = {"n", RAD_P_INT, n}; p[2].key = "dtype"; p[2].kind = RAD_P_STR; p[2].sval = "f32";
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 3;
        t[1].data = dev.data(); t[2].data = dval.data();
        int rc = halo_rad_pair_argmax_f32(&a, nullptr);
        t[1].data = hst.data(); t[2].data = hval.data();
        int rh = host_pair_argmax(&a);
        /* a pitched output: every third element, the rest must stay untouched */
        RadTensor tp = t[1]; tp.data = pitched.data(); tp.stride[0] = 3; RadTensor t3[3] = {t[0], tp, RadTensor{}};
        RadArgs ap = a; ap.t = t3;
        int rp = halo_rad_pair_argmax_f32(&ap, nullptr);
        bool ok = rc == 0 && rh == 0 && rp == 0;
        for (int r = 0; r < rows && ok; ++r) {
            ok = dev[r] == ref[r] && hst[r] == ref[r] && pitched[3 * r] == ref[r] && pitched[3 * r + 1] == 777 && pitched[3 * r + 2] == 777;
            if (ok && ref[r] < 0) ok = std::isinf(dval[r]) && dval[r] < 0;
        }
        ++cases; if (!ok) { ++fails; std::printf("FAIL world %d R %d rows %d rank%d\n", world, R, rows, rank3 ? 3 : 2); }
    }
    std::printf("pair_argmax: %d cases, %d failure(s)\n", cases, fails);
    return fails != 0;
}
