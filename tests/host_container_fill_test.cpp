/* Actual CPU container refill: clipped writes, bounded inputs and no GPU transfer objects. */
#include "rad_test.h"
#include "engine_priv.h"
#include <unistd.h>
namespace {
std::atomic<int> copies{0}, streams{0}, pinned{0};
int stream_create(RadStream* stream, int priority) { ++streams; return rad_stream_create(stream,priority); }
void* device_alloc(int64_t bytes,int kind) { if(kind==RAD_MEM_HOST_PINNED) ++pinned; return rad_dev_alloc(bytes,kind); }
int copy_async(void* dst,const void* src,int64_t bytes,RadStream stream) { ++copies; return rad_memcpy_async(dst,src,bytes,stream); }
}
#define rad_stream_create stream_create
#define rad_dev_alloc device_alloc
#define rad_memcpy_async copy_async
#define RAD_LOADER_INTERNALS_TEST 1
#include "../core/engine_bringup.cpp"
#undef RAD_LOADER_INTERNALS_TEST
#undef rad_stream_create
#undef rad_dev_alloc
#undef rad_memcpy_async
namespace rad::loader_test {
struct LoaderTransferTestAccess {
    static int run(RelayoutPool &pool, StagedWeight &weight, const uint8_t *input) {
        const uint8_t *planes[] = {input};
        return pool.run(weight, planes);
    }
};
}
TEST(cpu_refill_clips_identity_and_converted_writes_without_gpu_copies) {
    if (!rad::device_is_host()) {
        radtest::skip("host backend only");
        return;
    }
    const int64_t page = sysconf(_SC_PAGESIZE);
    void *mapping =
        mmap(nullptr, (size_t)(5 * page), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(mapping != MAP_FAILED);
    struct Unmap {
        void *p;
        int64_t n;
        ~Unmap() { munmap(p, (size_t)n); }
    } unmap{mapping, 5 * page};
    auto *dst = static_cast<uint8_t *>(mapping);
    REQUIRE_EQ(mprotect(dst + page, (size_t)page, PROT_READ | PROT_WRITE), 0);
    REQUIRE_EQ(mprotect(dst + 3 * page, (size_t)page, PROT_READ | PROT_WRITE), 0);
    std::vector<std::pair<uint8_t *, uint8_t *>> clips{{dst + page, dst + 2 * page},
                                                       {dst + 3 * page, dst + 4 * page}};
    for (bool convert : {false, true}) {
        rad::Program program;
        program.weights.resize(2);
        auto &weight = program.weights[1];
        weight.stored_bytes = 5 * page;
        weight.identity = true;
        weight.widen = convert;
        weight.stored_dtype = convert ? RAD_F32 : RAD_U8;
        weight.n_sel = 1;
        weight.sel[0] = 0;
        weight.sel_rows[0] = weight.stored_bytes / (convert ? 4 : 1);
        weight.sel_cols[0] = 1;
        weight.enc.n_planes = 1;
        weight.enc.plane[0].dtype = convert ? RAD_BF16 : RAD_U8;
        std::vector<uint8_t> input(
            (size_t)(convert ? weight.stored_bytes / 2 : weight.stored_bytes));
        for (int64_t i = 0; i < weight.sel_rows[0]; ++i)
            if (convert)
                rad_store_f32(input.data(), RAD_BF16, i, float(i % 127));
            else
                input[(size_t)i] = (uint8_t)(i % 251);
        std::memset(dst + page, 0xa5, (size_t)page);
        std::memset(dst + 3 * page, 0xa5, (size_t)page);
        std::vector<rad::loader_test::StagedWeight> staged(1);
        auto &item = staged[0];
        item.prog = &program;
        item.wi = 1;
        item.dst = dst;
        item.dev = false;
        item.clip = &clips;
        rad::loader_test::RelayoutPool pool(staged);
        copies = 0;
        CHECK_OK(rad::loader_test::LoaderTransferTestAccess::run(pool, item, input.data()));
        CHECK_EQ(copies.load(), 0);
        for (const auto &clip : clips) {
            for (auto *p = clip.first; p < clip.second; p += convert ? 4 : 1) {
                const int64_t off = p - dst;
                if (convert)
                    CHECK_EQ(rad_load_f32(p, RAD_F32, 0), float((off / 4) % 127));
                else
                    CHECK_EQ(*p, (uint8_t)(off % 251));
            }
        }
    }
}
TEST(host_fill_preflight_bounds_all_inputs_and_partial_output_before_mapping) {
    rad::Program program;
    program.weights.resize(2);
    auto &weight = program.weights[1];
    weight.identity = true;
    weight.stored_bytes = 4096;
    std::vector<uint8_t> memory(8192);
    std::vector<std::pair<uint8_t *, uint8_t *>> clips{{memory.data(), memory.data() + 4096}};
    rad::loader_test::StagedWeight item;
    item.prog = &program;
    item.wi = 1;
    item.dst = memory.data();
    item.clip = &clips;
    item.in_bytes = 2049;
    std::vector<rad::loader_test::StagedWeight> staged(2, item);
    CHECK_OK(rad::loader_test::RelayoutPool::check_host_fill(staged, 8192));
    staged.push_back(item);
    CHECK_EQ(rad::loader_test::RelayoutPool::check_host_fill(staged, 8192), RAD_E_FULL);
    staged.resize(1);
    staged[0].in_bytes = -1;
    CHECK_EQ(rad::loader_test::RelayoutPool::check_host_fill(staged, 8192), RAD_E_FULL);
    staged[0].in_bytes = INT64_MAX;
    CHECK_EQ(rad::loader_test::RelayoutPool::check_host_fill(staged, INT64_MAX), RAD_E_FULL);
    const int64_t reserve_bytes = ((int64_t)1 << 30) + 4096;
    void *large =
        mmap(nullptr, (size_t)reserve_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(large != MAP_FAILED);
    struct Unmap {
        void *p;
        int64_t n;
        ~Unmap() { munmap(p, (size_t)n); }
    } unmap{large, reserve_bytes};
    staged[0].in_bytes = 1;
    staged[0].dst = static_cast<uint8_t *>(large);
    clips = {{staged[0].dst, staged[0].dst + 4096}};
    weight.widen = true;
    weight.stored_bytes = ((int64_t)1 << 30) + 1;
    CHECK_EQ(rad::loader_test::RelayoutPool::check_host_fill(staged, 8192), RAD_E_FULL);
    CHECK_EQ(memory.front(), 0);
}
TEST(cpu_container_sweep_preserves_strides_and_clipped_layout_without_gpu_resources) {
    if (!rad::device_is_host()) {
        radtest::skip("host backend only");
        return;
    }
    const std::string path = radtest::tmp_path("cpu-fill.rad");
    struct Remove {
        const std::string &p;
        ~Remove() { std::remove(p.c_str()); }
    } remove{path};
    constexpr int64_t count = 256;
    std::vector<uint8_t> input((size_t)(count * 2));
    for (int64_t i = 0; i < count; ++i)
        rad_store_f32(input.data(), RAD_BF16, i, float(i));
    rad::RadWriter writer;
    REQUIRE_EQ(writer.begin(path.c_str()), RAD_OK);
    writer.set_arch("radtest");
    rad::RadWriter::Weight source;
    source.name = "source";
    source.enc = rad_enc_plain(RAD_BF16);
    source.rank = 2;
    source.shape[0] = count;
    source.shape[1] = 1;
    const void *planes[] = {input.data()};
    REQUIRE(writer.add_weight(source, planes) >= 0);
    REQUIRE_EQ(writer.finish(), RAD_OK);
    rad::RadFile file;
    REQUIRE_EQ(file.open(path.c_str()), RAD_OK);
    const auto *entry = file.find("source");
    REQUIRE(entry != nullptr);
    const int64_t offset = (int64_t)file.plane(*entry, 0).offset;
    rad::Program program;
    program.weights.resize(2);
    auto &weight = program.weights[1];
    weight.stored_bytes = count * 4;
    weight.identity = true;
    weight.widen = true;
    weight.stored_dtype = RAD_F32;
    weight.n_sel = 1;
    weight.sel[0] = 0;
    weight.sel_rows[0] = count;
    weight.sel_cols[0] = 1;
    weight.enc = source.enc;
    std::vector<uint8_t> output((size_t)weight.stored_bytes, 0xa5), verbatim(60, 0);
    std::vector<std::pair<uint8_t *, uint8_t *>> clips{{output.data() + 64, output.data() + 128},
                                                       {output.data() + 512, output.data() + 580}};
    std::vector<rad::loader_test::StagedWeight> staged(1);
    auto &item = staged[0];
    item.prog = &program;
    item.wi = 1;
    item.dst = output.data();
    item.clip = &clips;
    item.in_bytes = count * 2;
    item.remaining = count * 2;
    REQUIRE_EQ(rad::loader_test::RelayoutPool::check_host_fill(staged, 4096), RAD_OK);
    rad::loader_test::RelayoutPool pool(staged);
    pool.start(2, 4096, true);
    std::vector<rad::loader_test::LoadRun> runs{{offset, count * 2, nullptr, count * 2, count * 2, 1, 0, 0, 0},
                                   {offset + 4, 9 * 16 + 6, verbatim.data(), 6, 16, 10, 0, 0, -1}};
    copies = 0;
    streams = 0;
    pinned = 0;
    int64_t read = 0;
    int before = 0;
    CHECK_OK(rad::loader_test::sweep_container(&file, runs, {0}, &pool, &read, [&] {
        ++before;
        return RAD_OK;
    }));
    CHECK_OK(pool.finish());
    CHECK_EQ(before, 1);
    CHECK_EQ(copies.load(), 0);
    CHECK_EQ(streams.load(), 0);
    CHECK_EQ(pinned.load(), 0);
    CHECK(read >= count * 2);
    for (int64_t i = 0; i < weight.stored_bytes; ++i) {
        const bool written = (i >= 64 && i < 128) || (i >= 512 && i < 580);
        if (!written)
            CHECK_EQ(output[(size_t)i], 0xa5);
        else if (i % 4 == 0)
            CHECK_EQ(rad_load_f32(output.data() + i, RAD_F32, 0), float(i / 4));
    }
    for (int64_t row = 0; row < 10; ++row)
        for (int64_t col = 0; col < 6; ++col)
            CHECK_EQ(verbatim[(size_t)(row * 6 + col)], input[(size_t)(4 + row * 16 + col)]);
}
RAD_TEST_MAIN()
