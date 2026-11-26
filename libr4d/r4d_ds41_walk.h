/* r4d_ds41_walk.h -- the DSpark Markov walk over the whole vocabulary (op dspark_walk_step): the
 * launch and the shape hook; included by r4d_plugin.h inside its extern "C".
 *
 * Kernel in r4d_ds41_walk.hip, specs and row in r4d_ds41_walk.inc, operands in r4d_ds41_walk_shapes.inc. */
#ifndef R4D_DS41_WALK_H
#define R4D_DS41_WALK_H

/* A key's slot in the keys / prev planes, in f32: the u64 key and 8 bytes of padding, so a row of the gather
 * the plugin issues between steps is one whole 16-byte word (libr4d's all-gather moves whole words a row:
 * r4d_ar_gather_admit.h). deepseek41 declares its planes with the same width (arch/deepseek41/ds41_walk_geom.h). */
#define R4D_DS41_WALK_SLOT 4

int r4d_rad_dspark_walk_step(const RadArgs*, RadStream);
int r4d_shape_dspark_walk_step(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

#endif
