/* table_copy.h -- a container table copied into host memory (--ngram-placement ram): locked memory
 * for the host gather (made here), or pinned memory a card reads (allocated by the caller, filled
 * by table_copy_fill). Both share the read and the check.
 *
 * The locked copy uses huge pages rather than locking the mapping: the mapping costs a fault per
 * 4 KiB page and a TLB walk on nearly every scattered row, while a MADV_HUGEPAGE region covers
 * thousands of rows per page. It is locked on first touch (MLOCK_ONFAULT); filled by parallel
 * positional reads, dropping the container's pages behind so the table is not held twice; and
 * checked against the mapping before use, since a wrong offset would silently serve wrong rows.
 * A host kernel reads an address no file backs as plain memory (libavx's table_file). */
#pragma once
#include "radfile.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace rad {

struct TableCopy {
    void*  base  = nullptr;
    size_t bytes = 0;       /* the mapping's length: the table rounded up to 2 MiB */
};

/* Copy [src, src + n) of `file`'s mapping (file offset `foff`) into a new locked region. RAD_OK with
 * `out` filled, or an error with `why` and nothing held. */
int  table_copy_make(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     TableCopy* out, std::string* why);

/* table_copy_make's read and check, into caller-allocated [dst, dst + n): pinned memory a card
 * reads in place when a device kernel gathers the table. */
int  table_copy_fill(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     uint8_t* dst, std::string* why);

/* The check table_copy_fill ends with, on its own: samples of [buf, buf + n) against [src, src + n)
 * (the first and last 4 KiB and 64 spread between). RAD_OK, or RAD_E_IO with `why`. For a copy
 * another process filled. */
int  table_copy_check(const uint8_t* src, int64_t n, const uint8_t* buf, std::string* why);

/* Unlock and unmap; a no-op for a copy never made. */
void table_copy_free(TableCopy* c);

}  /* namespace rad */
