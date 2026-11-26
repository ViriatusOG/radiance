/* iface.h -- what the tools need from components outside tools/.
 *
 * rad-info, rad-convert, rad-kbench and rad-tune build on the GGUF reader and tokenizer.json parser
 * (core/text) and the plugin registry and declare phase (core/plugin, core/build). This header is
 * the tools' single seam onto them.
 *
 * rad-convert, rad-kbench and rad-tune are the engine's startup with a different last step: plugins
 * loaded in hierarchy order and the declare phase run against a model's metadata (spec §4.4, §15,
 * §17).
 */
#pragma once
#include "rad_core.h"
#include "format/radfile.h"

#include "text/gguf.h"
#include "build/startup.h"

#include <string>
#include <vector>

namespace rad {

/* The vocab pipeline (VocabBuild, parse_tokenizer_json, vocab_from_gguf, vocab_serialize) is in
 * text/tokenizer.h; the container side is RadWriter::intern_string() and set_vocab_section().
 * Plugin loading and the declare phase are in build/startup.h. */

/* ================================================================== shared CLI plumbing */
/* Each tool parses its own flags; these are the few things they must agree on. */

std::string rad_home();                              /* $RADIANCE_HOME, else the install prefix */
std::vector<std::string> rad_hierarchy_from_env();   /* $RADIANCE_KERNELS */
/* A kernel list as --kernels and $RADIANCE_KERNELS spell it: ',' or ':' separated, the same rule
 * the engine reads them by (util/kernel_list.h). */
std::vector<std::string> rad_split_list(const std::string& s);

/* `text` as a whole decimal number in [lo, hi]: the engine's rule for a numeric flag
 * (core/config.cpp), so "12k", "8x", "", "auto" and an overflow are refused rather than read as
 * their leading digits or as 0. False with the reason in *why. */
bool rad_parse_int(const char* text, int64_t lo, int64_t hi, int64_t* out, std::string* why);

/* rad_parse_int for a flag's value: the number, or a message naming `tool` and `flag` on stderr
 * and exit status 2. */
int64_t rad_flag_int(const char* tool, const char* flag, const char* text, int64_t lo, int64_t hi);

/* Loads the quantizer plugins into `reg` from the `quantizers/` directory of each directory of
 * `home` that has one, or of the first when none does. Negative when a plugin fails to load. */
int rad_load_quantizers(Registry& reg, const std::string& home);

/* A container's RadModelMeta, as the declare phase takes it. The meta's key/value arrays point
 * into this object, so it stays where it is while the meta is in use. */
struct ContainerMeta {
    RadModelMeta meta{};
    explicit ContainerMeta(const RadFile& f);
    ContainerMeta(const ContainerMeta&) = delete;
    ContainerMeta& operator=(const ContainerMeta&) = delete;

private:
    std::vector<std::string> keys_, vals_;
    std::vector<const char*> key_ptrs_, val_ptrs_;
};

/* The value after the flag at argv[*i], advancing *i past it; a message naming `tool` and `flag`
 * on stderr and exit status 2 when the command line ends there. */
const char* rad_flag_value(const char* tool, int argc, char** argv, int* i, const char* flag);

}  /* namespace rad */
