/* bailing_hybrid_gguf -- the GGUF variant of bailing_hybrid for the tiny family's
 * IQ4_NL release: the same declaration and step, claiming the ftype-25 descriptor so the
 * quantised GGUF converts and serves. As in hyv3_gguf, a thin include over the family's
 * implementation with the namespace, the plugin name and the quant redefined. */
#define bailing_hybrid bailing_hybrid_gguf
#define BAILING_HYBRID_PLUGIN "bailing_hybrid_gguf"
#define BAILING_HYBRID_QUANT "gguf_ftype_25"
#include "../bailing_hybrid/bailing_hybrid.cpp"

/* GGUF file_type names the quant, never an imported tool descriptor: imports stay
 * with the family's bf16 plugin (rad_builder.h, rad_arch_serves_imports). */
RAD_ARCH_SERVES_IMPORTS(0)
