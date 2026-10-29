/* gguf_writer.h -- GGUF v3 files written by tests: header keys and tensors, the tensors' bytes as
 * the caller built them. */
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ggufw {

struct GKv {
    std::string key;
    uint32_t type;                     /* 4 u32, 6 f32, 8 string, 9 array of strings */
    double num = 0;
    std::vector<std::string> strs;
};
struct GTensor {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> ne;          /* fastest dimension first, as GGUF stores it */
    std::vector<uint8_t> bytes;
};

inline void put_bytes(std::string* b, const void* p, size_t n) { b->append((const char*)p, n); }
inline void put_str(std::string* b, const std::string& s) {
    const uint64_t n = s.size();
    put_bytes(b, &n, 8);
    b->append(s);
}

/* A GGUF v3 file of `kvs` and `ts`, every tensor at a 32-byte aligned offset. */
inline bool write_gguf(const std::string& path, const std::vector<GKv>& kvs,
                       const std::vector<GTensor>& ts) {
    std::string b = "GGUF";
    const uint32_t ver = 3;
    const uint64_t nt = ts.size(), nk = kvs.size();
    put_bytes(&b, &ver, 4);
    put_bytes(&b, &nt, 8);
    put_bytes(&b, &nk, 8);
    for (const GKv& kv : kvs) {
        put_str(&b, kv.key);
        put_bytes(&b, &kv.type, 4);
        if (kv.type == 4) { const uint32_t v = (uint32_t)kv.num; put_bytes(&b, &v, 4); }
        else if (kv.type == 6) { const float v = (float)kv.num; put_bytes(&b, &v, 4); }
        else if (kv.type == 8) put_str(&b, kv.strs.at(0));
        else {
            const uint32_t et = 8;
            const uint64_t n = kv.strs.size();
            put_bytes(&b, &et, 4);
            put_bytes(&b, &n, 8);
            for (const std::string& s : kv.strs) put_str(&b, s);
        }
    }
    uint64_t off = 0;
    for (const GTensor& t : ts) {
        put_str(&b, t.name);
        const uint32_t nd = (uint32_t)t.ne.size();
        put_bytes(&b, &nd, 4);
        for (uint64_t d : t.ne) put_bytes(&b, &d, 8);
        put_bytes(&b, &t.type, 4);
        put_bytes(&b, &off, 8);
        off += (t.bytes.size() + 31) / 32 * 32;
    }
    while (b.size() % 32) b.push_back('\0');
    for (const GTensor& t : ts) {
        b.append((const char*)t.bytes.data(), t.bytes.size());
        while (b.size() % 32) b.push_back('\0');
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    return std::fclose(f) == 0 && ok;
}

}  /* namespace ggufw */
