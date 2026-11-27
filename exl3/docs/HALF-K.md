# Half-integer K (exllamav3 1.6): K = 1.5, 2.5, 3.5

exllamav3 1.6 quantises at half-integer bitrates (mul1 codebook only). turboderp's DeepSeek-V4.1-Flash-1017
conversions use them: 2.25bpw has its routed experts at 2.5 bits in layers 0-5 and 35-39 (2 bits elsewhere),
2.68bpw at 2.5 / 3, 1.68bpw at 1.5 / 2.

## The format (exllamav3 1.6.5: `exl3_lib/quantize.py` frac_k, `exllamav3_ext/quant/frac.cu`, `exl3_dq.cuh` dq8_half)

- A 16 x 16 tile is a ring of 16 K uint16 = 128 * K2 bits (K2 = 2 K; 40 uint16 at 2.5), u32 words little-endian,
  MSB-first: the same stream convention as integer K.
- Position p takes D(p) = KA + bit (p mod 16) of 0xAAAA bits: the ODD positions carry the extra bit. Its symbol is
  stream bits [S(p - 1), S(p)), S(p) = floor((p + 1) K2 / 2); its 16-bit state is the window ending at S(p),
  circular in the ring. At integer K, S(p) = (p + 1) K: one formula for both.
- Positions map to tile elements exactly as at integer K (`frac_perm` is `tensor_core_perm`).

## How this extension carries it

- **A tagged `bits`.** Every `int bits` in common/, include/, the quantiser and both kernel libraries takes K tagged:
  integer K is itself (1..8), half-integer K is `EXL3_HALF(KA) = 16 + 2 KA + 1` (1.5 -> 19, 2.5 -> 21, 3.5 -> 23).
  `exl3_k2(bits)` is 2 K, `exl3_sym_off(bits, j) = floor(j K2 / 2)`, `exl3_sym_width(bits, j)` the difference.
  Integer K paths are unchanged; the half-K state is read off the packed ring (a second method; the integer loop
  stays the oracle).
- **The encoding.** A uniform uK plane cannot hold symbols of two widths, so the codes plane is
  `u(2KA+1)` TILED **[1 x 2]**: element (i, m) of row i (output feature n = i) is the PAIR symbol of columns 2m
  (input feature k = 2m, KA bits, low) and 2m + 1 (KA + 1 bits, high). Positions 2m and 2m + 1 of a tile are tile rows
  r and r + 1 (k axis, r even) of one column, so this is a lossless permutation of the ring, and reading column j at
  bit off(j) = floor(j K2 / 2) of the row gives the same bits as reading the pair plane. 2.5: u5 [1x2],
  `exl3:codes=u5[1x2],suh=f16[*x1],svh=f16[1x*][,suh.1=f16[*x1]]/mul1/h128`. Planes per row: cols / 2 elements.
- **Names.** GEMM dtype strings `exl3k1h`, `exl3k2h`, `exl3k3h`. Recipe: `exl3 src=DIR bits=2.5 codebook=mul1`
  (the quantiser's `bits` option is RAD_P_F64; `bits=4` reads as before). A half-integer K in another codebook is
  refused at the encoding, the quantiser and the geometry parse (RAD_E_DTYPE).
- **exl3r4d.** The stored form is the trellis record as for integer K (`exl3r4d.trellis.k2h`, 80 bytes a tile);
  `Plan` and `decode_run` work in half-bit units (a tile 4 K2 words, a run of four positions from an even q0 ends at
  (q0 / 2 + 2) K2, states at 3KA+2 / 2KA+1 / KA+1 / 0 bits below it, as dq8_half). Built for K = 2.5 only (the rows'
  dtype list carries `exl3k2h`), in the decode band's generic K path (kx_fn, DNW) and every M > 16 form (pair, map,
  grp, fused). The fused gate_up_down launch stays K = 4's.
- Versions: EXL3_ENCODING_VERSION 2, EXL3_DECODE_API_VERSION 2, EXL3_LAYOUT_API_VERSION 2.

## Verified without a GPU

- common: hand-built tiles (stream bits and a state across the ring's wrap), every half K's symbols / pack round
  trip, states against the integer oracle's backward-concatenation method, the pair-plane bit identity, W_hat from
  the plane == W_hat from the trellis, and a negative control (the same bytes read as integer K disagree).
- layout: canonical <-> record round trips and TP slices at 1.5 / 2.5 / 3.5 on the real expert shapes; the device
  library's fast relayout against the oracle (and DeepSeek's [2304, 5120] / [5120, 2304]); its hooks claim the pair
  plane, refuse a [rows, cols] or u4 codes plane, and an op of the other K.
- exl3ref through the op ABI (exl3ref_cases): gate_up and down at 1.5 / 2.5 / 3.5, two classes and whole experts,
  against the numpy transcription (tests/ref/exl3_numpy.py, extended): max_ulp 0.
- Real tensors (tools/half_k): eight of turboderp's 2.5-bit DeepSeek expert tensors (layers 0, 3, 35, 37; 94.4 M
  weights) decoded by the C oracle and by an independent numpy transcription of exllamav3's frac decode: 0
  mismatches; the other parity (0x5555) disagrees on about half the weights. The pair plane round trip and the
  device relayout give back the source trellis byte for byte.

## Not verified (needs a GPU)

The device decode at K = 2.5: `tests/gpu/test_exl3r4d.py --suite device` runs it (bits_band at 2.5, and
deepseek_ep: DeepSeek's whole-expert shapes at K = 2 and 2.5, M = 1 / 16 / 64, with the h_rot and prep_gu chains).
