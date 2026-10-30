# IQR codebooks

The integer codebooks the IQR recipes name. Every value is an int8 (-127..127), so a weight
quantised against one is on the int8 grid: VRAM holds its codes at their own width, libr4d's IQR
rows decode each to its int8 value where the weight is read, and the trunk runs on the i8a8h GEMMs
with rotated int8 activations (docs/IQR.md). A container carries the table its weights were
quantised against, so serving needs none of these files; quantising with a recipe does.

| file | what | made from |
|---|---|---|
| `iqr-scalar.txt` | the scalar Lloyd codebooks at 2..6 bits, one line `<bits> <values;...>` (recipes inline them as `table=`) | Lloyd-Max on Qwen3.8-27B layer 3's weights rotated by fwht128 and normalised by each group-of-128's absmax |
| `iqr-vq2-4x256.txt` | 256 codewords of 4 (2 bits a weight), `vq=` | k-means on the same weights normalised by each group's rms; `# unit` is the integer span of a unit-rms vector |
| `iqr-vq225-4x512.txt` | 512 codewords of 4 (2.25 bits a weight), `vq=` | the same, 512 centroids |
| `iqr-vq25-4x1024.txt` | 1024 codewords of 4 (2.5 bits a weight), `vq=` | the same, 1024 centroids |
| `iqr-vq3-4x4096.txt` | 4096 codewords of 4 (3 bits a weight), `vq=` | the same, 4096 centroids |
| `iqr-tcq-L12-k2.txt` .. `-k7.txt` | the 4096 state values of a 12-bit tail-biting trellis over 2..7-bit codes, `tcq=12 tcq_lut=` | Lloyd iterations on the Viterbi paths down the columns of the same weights normalised by each group's rms, from shuffled normal quantiles; `# unit` as above |
| `iqr-tcq-L14-k2.txt` .. `-k4.txt` | the 16384 state values of a 14-bit trellis over 2..4-bit codes, `tcq=14 tcq_lut=` | the same |

Held-out mean squared error per weight of unit-rms rotated weights: scalar 2-bit 0.1159 against the
256 x 4 codebook's 0.0965; the 512 x 4 one 0.0697, the 1024 x 4 one 0.0500; scalar 3-bit 0.0342 against the 4096 x 4 codebook's 0.0256.

The trellis tables, against the default shuffled normal quantiles (unit-rms rotated weights, in
sample): 2 bits 0.0733 to 0.0717, 3 bits 0.0197 to 0.0188, 4 bits 0.00529 to 0.00497, 5 bits
0.00146 to 0.00133, 6 bits 0.00043 to 0.00037 (Lloyd-Max scalar at 2 bits: 0.1175). The 14-bit
tables: 2 bits 0.0694, 3 bits 0.0179, 4 bits 0.00466.
