/* torch_zip_test.cpp -- a PyTorch checkpoint read as data (core/format/torch_zip.h).
 *
 * Each case writes a torch.save archive the way torch lays one out (stored zip entries, a protocol
 * 2 pickle of the state dict naming its storages by persistent id) with a small writer of the same
 * subset, then reads it back: the tensors' names, dtypes, shapes and bytes, a nested dict, a
 * parameter, a ZIP64 directory. The refusals are the point as much as the reads: a global outside
 * the weights-only allow-list (the shape every malicious pickle has), an opcode it does not use, a
 * compressed or truncated entry and a tensor its storage does not hold are each refused by name,
 * with nothing run.
 *
 * RAD_TEST_TORCH_PTH, when set, names a real checkpoint (Fish Audio's codec.pth) that is opened as
 * well; the case is skipped without it.
 */
#include "rad_test.h"
#include "format/torch_zip.h"
#include "format/checkpoint.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

/* ------------------------------------------------------------------ the pickle */
struct Pkl {
    std::string b;
    void op(char c) { b += c; }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) b += (char)((v >> (8 * i)) & 0xff); }
    void str(const std::string& s) { op('X'); u32((uint32_t)s.size()); b += s; }
    void global(const std::string& mod, const std::string& name) { op('c'); b += mod + "\n" + name + "\n"; }
    void integer(int64_t v) {
        if (v >= 0 && v < 256) { op('K'); b += (char)v; }
        else { op('J'); u32((uint32_t)(int32_t)v); }
    }
    void tuple_of(const std::vector<int64_t>& v) { op('('); for (int64_t x : v) integer(x); op('t'); }
    void ordered_dict() { global("collections", "OrderedDict"); op(')'); op('R'); }
};

struct Tensor {
    std::string name, storage_cls, key;
    std::vector<int64_t> shape, stride;
    int64_t offset = 0, numel = 0;   /* the storage's elements */
    bool parameter = false;
};

/* One tensor's _rebuild_tensor_v2 call, optionally wrapped as a parameter. */
void put_tensor(Pkl& p, const Tensor& t, const std::string& storage_mod = "torch") {
    if (t.parameter) p.global("torch._utils", "_rebuild_parameter");
    p.global("torch._utils", "_rebuild_tensor_v2");
    p.op('(');
    p.op('(');
    p.str("storage");
    p.global(storage_mod, t.storage_cls);
    p.str(t.key);
    p.str("cpu");
    p.integer(t.numel);
    p.op('t');
    p.op('Q');
    p.integer(t.offset);
    p.tuple_of(t.shape);
    p.tuple_of(t.stride);
    p.op((char)0x89);   /* requires_grad False */
    p.ordered_dict();
    p.op('t');
    p.op('R');
    if (t.parameter) {
        /* (tensor, requires_grad, hooks): the tensor is below; build the argument tuple around it. */
        p.op((char)0x89);
        p.ordered_dict();
        p.op((char)0x87);   /* TUPLE3 */
        p.op('R');
    }
}

/* A state dict of `ts`, and `nested` under "generator." as a second dict. */
std::string state_dict(const std::vector<Tensor>& ts, const std::vector<Tensor>& nested = {}) {
    Pkl p;
    p.op((char)0x80);
    p.b += (char)2;
    p.ordered_dict();
    p.op('(');
    for (const Tensor& t : ts) {
        p.str(t.name);
        put_tensor(p, t);
    }
    if (!nested.empty()) {
        p.str("generator");
        p.ordered_dict();
        p.op('(');
        for (const Tensor& t : nested) {
            p.str(t.name);
            put_tensor(p, t);
        }
        p.op('u');
    }
    p.str("epoch");          /* a value that is not a tensor: skipped */
    p.integer(7);
    p.op('u');
    /* The state dict's _metadata, as torch writes it: BUILD with a dict state. */
    p.op('}');
    p.op('b');
    p.op('.');
    return p.b;
}

/* ------------------------------------------------------------------ the zip */
struct ZipEntry { std::string name, data; uint16_t method = 0; };

void le(std::string* s, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) s->push_back((char)((v >> (8 * i)) & 0xff));
}

/* Stored entries, each local header's extra field padded so the data starts 64-byte aligned (as
 * torch.save aligns storages), then the central directory, ZIP64 records when asked. */
std::string zip_of(const std::vector<ZipEntry>& es, bool zip64 = false) {
    std::string out, cd;
    for (const ZipEntry& e : es) {
        const uint64_t off = out.size();
        const uint64_t base = off + 30 + e.name.size() + 4;
        const uint64_t pad = (64 - base % 64) % 64;
        le(&out, 0x04034b50, 4); le(&out, 20, 2); le(&out, 0, 2); le(&out, e.method, 2);
        le(&out, 0, 4); le(&out, 0, 4);
        le(&out, e.data.size(), 4); le(&out, e.data.size(), 4);
        le(&out, e.name.size(), 2); le(&out, 4 + pad, 2);
        out += e.name;
        le(&out, 0xcafe, 2); le(&out, pad, 2); out.append((size_t)pad, '\0');
        out += e.data;
        std::string ex;
        uint32_t loff32 = (uint32_t)off;
        if (zip64) {
            le(&ex, 1, 2); le(&ex, 8, 2); le(&ex, off, 8);
            loff32 = 0xffffffffu;
        }
        le(&cd, 0x02014b50, 4); le(&cd, 45, 2); le(&cd, 45, 2); le(&cd, 0, 2); le(&cd, e.method, 2);
        le(&cd, 0, 4); le(&cd, 0, 4);
        le(&cd, e.data.size(), 4); le(&cd, e.data.size(), 4);
        le(&cd, e.name.size(), 2); le(&cd, ex.size(), 2); le(&cd, 0, 2);
        le(&cd, 0, 2); le(&cd, 0, 2); le(&cd, 0, 4); le(&cd, loff32, 4);
        cd += e.name + ex;
    }
    const uint64_t cd_off = out.size();
    out += cd;
    if (zip64) {
        const uint64_t z = out.size();
        le(&out, 0x06064b50, 4); le(&out, 44, 8); le(&out, 45, 2); le(&out, 45, 2);
        le(&out, 0, 4); le(&out, 0, 4); le(&out, es.size(), 8); le(&out, es.size(), 8);
        le(&out, cd.size(), 8); le(&out, cd_off, 8);
        le(&out, 0x07064b50, 4); le(&out, 0, 4); le(&out, z, 8); le(&out, 1, 4);
    }
    le(&out, 0x06054b50, 4); le(&out, 0, 2); le(&out, 0, 2);
    le(&out, zip64 ? 0xffff : es.size(), 2); le(&out, zip64 ? 0xffff : es.size(), 2);
    le(&out, zip64 ? 0xffffffffu : cd.size(), 4); le(&out, zip64 ? 0xffffffffu : cd_off, 4);
    le(&out, 0, 2);
    return out;
}

std::string floats(std::initializer_list<float> v) {
    std::string s(v.size() * 4, '\0');
    std::memcpy(s.data(), v.begin(), s.size());
    return s;
}

/* The archive of `pkl` with the storages "0" and "1": 6 and 4 floats. */
std::string archive(const std::string& pkl, bool zip64 = false, uint16_t method = 0) {
    return zip_of({ { "ck/data.pkl", pkl, method }, { "ck/byteorder", "little" },
                    { "ck/data/0", floats({ 1, 2, 3, 4, 5, 6 }) },
                    { "ck/data/1", floats({ 7, 8, 9, 10 }) } },
                  zip64);
}

std::vector<Tensor> two() {
    return { Tensor{ "a.weight", "FloatStorage", "0", { 2, 3 }, { 3, 1 }, 0, 6 },
             Tensor{ "a.bias", "FloatStorage", "1", { 2 }, { 1 }, 2, 4, true } };
}

const TorchTensor* by_name(const TorchZipFile& f, const std::string& n) {
    for (const TorchTensor& t : f.tensors())
        if (t.name == n) return &t;
    return nullptr;
}

float f32_at(const TorchTensor& t, int i) {
    float v;
    std::memcpy(&v, t.data + 4 * i, 4);
    return v;
}

}  /* namespace */

/* The tensors as torch.save wrote them: names, dtype, shape, the bytes at the storage offset, a
 * parameter's tensor, a nested dict's names dotted and an integer skipped; in a plain and a ZIP64
 * directory. */
TEST(a_state_dict_reads_back_as_its_tensors) {
    for (bool z64 : { false, true }) {
        const std::string bytes = archive(state_dict(two(), { Tensor{ "x", "FloatStorage", "0",
                                                                      { 3 }, { 1 }, 3, 6 } }),
                                          z64);
        TorchZipFile f;
        std::string why;
        REQUIRE_EQ(f.open_bytes((const uint8_t*)bytes.data(), bytes.size(), &why), (int)RAD_OK);
        CHECK_EQ(f.tensors().size(), (size_t)3);
        const TorchTensor* w = by_name(f, "a.weight");
        REQUIRE(w != nullptr);
        CHECK_EQ(w->dtype, (uint32_t)RAD_F32);
        CHECK(w->shape == std::vector<int64_t>({ 2, 3 }));
        CHECK_EQ(w->bytes, (uint64_t)24);
        CHECK_EQ(f32_at(*w, 5), 6.0f);
        CHECK_EQ((uintptr_t)w->data % 64, (uintptr_t)bytes.data() % 64);   /* mapped in place */
        const TorchTensor* bias = by_name(f, "a.bias");
        REQUIRE(bias != nullptr);
        CHECK_EQ(f32_at(*bias, 0), 9.0f);        /* storage 1 from element 2 */
        CHECK_EQ(f32_at(*bias, 1), 10.0f);
        const TorchTensor* x = by_name(f, "generator.x");
        REQUIRE(x != nullptr);
        CHECK_EQ(f32_at(*x, 0), 4.0f);
    }
}

/* What a malicious or foreign file holds is refused by name before anything is read past it: a
 * global outside the allow-list (os.system, builtins.eval), a storage class that is not one, an
 * opcode the subset does not use, a compressed entry, a pickle cut short, and a tensor that runs
 * past its storage or is not contiguous in it. */
TEST(anything_outside_the_weights_only_subset_is_refused_by_name) {
    auto refused = [](const std::string& bytes, const char* want) {
        TorchZipFile f;
        std::string why;
        const int st = f.open_bytes((const uint8_t*)bytes.data(), bytes.size(), &why);
        CHECK(st < 0);
        CHECK(has(why, want));
        if (!has(why, want)) std::fprintf(stderr, "    want %s, why: %s\n", want, why.c_str());
    };
    {
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.global("os", "system");
        p.op('('); p.str("echo pwned"); p.op('t'); p.op('R'); p.op('.');
        refused(archive(p.b), "'os.system'");
    }
    {
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.str("builtins"); p.str("eval"); p.op((char)0x93);
        p.op('.');
        refused(archive(p.b), "'builtins.eval'");
    }
    {
        std::vector<Tensor> ts = two();
        ts[0].storage_cls = "DoubleStorage";
        refused(archive(state_dict(ts)), "torch.DoubleStorage");
    }
    {
        std::string pk = state_dict(two());
        pk.insert(pk.size() - 1, 1, 'i');   /* INST, an old way to build any object */
        refused(archive(pk), "opcode 0x69");
    }
    refused(archive(state_dict(two()), false, 8), "compressed");
    {
        const std::string pk = state_dict(two());
        refused(archive(pk.substr(0, pk.size() / 2)), "pickle");
    }
    {
        std::vector<Tensor> ts = two();
        ts[1].shape = { 3 };                 /* three floats from element 2 of a 4-float storage */
        refused(archive(state_dict(ts)), "does not fit its storage");
    }
    {
        std::vector<Tensor> ts = two();
        ts[0].stride = { 1, 2 };             /* a transposed view */
        refused(archive(state_dict(ts)), "not contiguous");
    }
    refused(zip_of({ { "data.pkl", state_dict(two()) } }), "no <archive>/data.pkl");
    refused(std::string("PK\x03\x04 is a local header, and this is not an archive"),
            "end-of-central-directory");
}

/* Pickles that are well-formed but hostile to the reader: a dict that holds itself (the memo makes
 * one in a few bytes, and a recursive walk would never end), a dict shared by two parents, and a
 * shape whose element count wraps 64 bits to one the storage holds. */
TEST(a_cyclic_dict_and_a_wrapping_shape_are_refused) {
    auto refused = [](const std::string& pkl, const char* want) {
        const std::string bytes = archive(pkl);
        TorchZipFile f;
        std::string why;
        const int st = f.open_bytes((const uint8_t*)bytes.data(), bytes.size(), &why);
        CHECK(st < 0);
        CHECK(has(why, want));
        if (!has(why, want)) std::fprintf(stderr, "    want %s, why: %s\n", want, why.c_str());
    };
    {
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.op('}'); p.op('q'); p.b += (char)0;
        p.str("self"); p.op('h'); p.b += (char)0; p.op('s');
        p.op('.');
        refused(p.b, "reached twice");
    }
    {
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.op('}');
        p.str("a"); p.op('}'); p.op('q'); p.b += (char)1; p.op('s');
        p.str("b"); p.op('h'); p.b += (char)1; p.op('s');
        p.op('.');
        refused(p.b, "reached twice");
    }
    {
        /* (2^62 + 1, 4) at strides (4, 1): the product wraps to 4, which storage "0" holds. */
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.ordered_dict();
        p.op('(');
        p.str("big");
        p.global("torch._utils", "_rebuild_tensor_v2");
        p.op('(');
        p.op('('); p.str("storage"); p.global("torch", "FloatStorage"); p.str("0"); p.str("cpu");
        p.integer(6); p.op('t'); p.op('Q');
        p.integer(0);
        p.op('(');
        p.op((char)0x8a); p.b += (char)8;
        le(&p.b, ((uint64_t)1 << 62) + 1, 8);
        p.integer(4);
        p.op('t');
        p.tuple_of({ 4, 1 });
        p.op((char)0x89); p.ordered_dict(); p.op('t'); p.op('R');
        p.op('u');
        p.op('.');
        refused(p.b, "past 64 bits");
    }
}

/* Malformed archives, made by cutting a valid one at every length and by mutating its bytes, are
 * refused or read without a read outside the buffer: every tensor accepted lies inside it and
 * holds exactly the bytes its shape names. The buffer is an exact-size heap block, so a sanitiser
 * build sees any overrun. */
TEST(malformed_archives_are_refused_or_read_inside_the_file) {
    const std::vector<std::string> seeds = { archive(state_dict(two())),
                                             archive(state_dict(two(), two()), true) };
    int64_t accepted = 0, refused = 0;
    auto run = [&](const std::string& s) {
        std::vector<uint8_t> buf(s.begin(), s.end());
        TorchZipFile f;
        std::string why;
        if (f.open_bytes(buf.data(), buf.size(), &why) < 0) {
            ++refused;
            return;
        }
        ++accepted;
        for (const TorchTensor& t : f.tensors()) {
            uint64_t n = (uint64_t)rad_dtype_bits(t.dtype) / 8;
            for (int64_t d : t.shape) n *= (uint64_t)d;
            CHECK_EQ(n, t.bytes);
            CHECK(t.data >= buf.data() && t.data <= buf.data() + buf.size() &&
                  t.bytes <= (uint64_t)(buf.data() + buf.size() - t.data));
        }
    };
    for (const std::string& s : seeds)
        for (size_t n = 0; n < s.size(); ++n) run(s.substr(0, n));
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto next = [&]() {   /* splitmix64: a fixed sequence, so a failure reproduces */
        uint64_t z = (x += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    for (int it = 0; it < 20000; ++it) {
        std::string s = seeds[next() % seeds.size()];
        for (int k = 1 + (int)(next() % 8); k-- > 0;) {
            const size_t at = next() % s.size();
            static const uint8_t kEdge[] = { 0x00, 0x01, 0x7f, 0x80, 0xff };
            switch (next() % 3) {
            case 0: s[at] = (char)next(); break;
            case 1: s[at] = (char)(s[at] ^ (1 << (next() % 8))); break;
            default: s[at] = (char)kEdge[next() % 5]; break;
            }
        }
        run(s);
    }
    std::fprintf(stderr, "    %lld refused, %lld read\n", (long long)refused, (long long)accepted);
    CHECK(refused > 0);
    CHECK(accepted > 0);
}

/* A .pth beside the safetensors joins the checkpoint under its stem, so a plugin maps
 * `codec.<name>`; a torch copy of the model's own weights is passed over, and a file the reader
 * refuses is left out with a warning rather than failing the checkpoint. */
TEST(a_pth_beside_the_safetensors_joins_the_checkpoint_under_its_stem) {
    const std::string d = radtest::make_tmp_dir("tz");
    auto write = [&](const std::string& name, const std::string& bytes) {
        FILE* f = std::fopen((d + "/" + name).c_str(), "wb");
        REQUIRE(f != nullptr);
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    };
    /* A one-tensor safetensors model and its config. */
    {
        std::string hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        while (hdr.size() % 8) hdr += ' ';
        std::string st;
        le(&st, hdr.size(), 8);
        st += hdr + floats({ 1 });
        write("model.safetensors", st);
        write("config.json", "{\"model_type\": \"tiny\"}");
    }
    write("codec.pth", archive(state_dict(two())));
    write("pytorch_model.pth", archive(state_dict(two())));
    {
        Pkl p;
        p.op((char)0x80); p.b += (char)2;
        p.global("os", "system");
        p.op('.');
        write("optimizer.pth", archive(p.b));
    }
    Checkpoint ck;
    REQUIRE_EQ(ck.open(d), (int)RAD_OK);
    CHECK(ck.find("w") != nullptr);
    const SrcTensor* w = ck.find("codec.a.weight");
    REQUIRE(w != nullptr);
    CHECK_EQ(w->rad_dtype, (uint32_t)RAD_F32);
    CHECK(w->shape == std::vector<int64_t>({ 2, 3 }));
    CHECK(ck.find("codec.a.bias") != nullptr);
    CHECK(ck.find("pytorch_model.a.weight") == nullptr);
    CHECK(ck.find("optimizer.a.weight") == nullptr);
    std::filesystem::remove_all(d);
}

/* A real torch.save checkpoint, when one is named: every tensor's bytes inside the file. */
TEST(a_real_checkpoint_named_by_the_environment_reads) {
    const char* p = std::getenv("RAD_TEST_TORCH_PTH");
    if (!p || !*p) SKIP("RAD_TEST_TORCH_PTH names no checkpoint");
    TorchZipFile f;
    std::string why;
    REQUIRE_EQ(f.open(p, &why), (int)RAD_OK);
    CHECK(!f.tensors().empty());
    uint64_t bytes = 0;
    for (const TorchTensor& t : f.tensors()) bytes += t.bytes;
    std::fprintf(stderr, "    %s: %zu tensors, %.1f MiB\n", p, f.tensors().size(),
                 (double)bytes / (1 << 20));
}

RAD_TEST_MAIN()
