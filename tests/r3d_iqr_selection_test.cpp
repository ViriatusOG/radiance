#include "rad_test.h"
#include "build/rad_build.h"
#include "r3d_iqr.h"

namespace {
constexpr int columns = 256, reduction = 384, groups = reduction / 128;

void declare(rad::Registry& registry, const RadEncoding& encoding, bool factored) {
    const bool packed = rad_enc_is(&encoding, "trellis");
    const bool plain = rad_enc_is(&encoding, "plain");
    RadBuildCtx ctx{};
    ctx.world_size = 1;
    ctx.max_tok = 2048;
    rad::Builder builder(registry, RadModelMeta{}, ctx);
    builder.set_instances(false);
    builder.set_sources([&](const std::string& name, const rad::Program&, rad::WeightSource* out) -> int {
        if (name != "w" && !(plain && name == "scale")) return RAD_E_NOTFOUND;
        out->enc = name == "w" ? encoding : rad_enc_plain(RAD_BF16);
        out->rank = 2;
        out->shape[0] = columns;
        out->shape[1] = name == "w" ? reduction : groups;
        return RAD_OK;
    });
    RadWeightDecl decl{};
    decl.rank = 2;
    decl.shape[0] = columns;
    decl.shape[1] = reduction;
    decl.dtype = encoding.plane[0].dtype;
    decl.planes = packed ? "codes,trellis,trellis.mad" : "codes";
    decl.access = RAD_ACCESS_PER_TOKEN;
    decl.group = rad_group_layer(0);
    const rad_weight weight = builder.decl_weight("w", &decl);
    decl.dtype = factored ? RAD_F32 : RAD_BF16;
    decl.shape[1] = factored ? reduction : groups;
    decl.source = plain ? nullptr : "w";
    decl.planes = plain ? "codes" : factored ? "scale,scale.col" : "scale";
    const rad_weight scale = builder.decl_weight("scale", &decl);
    REQUIRE(weight && scale);
    const rad_weight bindings[]{weight, scale};
    const RadParam params[]{RAD_RANGE("M", 1, 2048), RAD_INT("N", columns), RAD_INT("K", reduction),
                            RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
    const auto handle = builder.decl_op("gemm_nt_q", params, 5, bindings, 2);
    REQUIRE(handle);
    const auto& bands = builder.program().op(handle)->bands;
    REQUIRE(!bands.empty());
    CHECK_EQ(bands.back().hi, 2048);
    for (const auto& band : bands) {
        const auto& resolved = band.dom[RAD_DOMAIN_DEVICE];
        REQUIRE(resolved);
        const auto* row = resolved.row->info;
        REQUIRE_EQ(std::string(row->name), packed ? "gemm_nt_q_iqr" : "gemm_nt_q_i8a8");
        REQUIRE(row->opd_shape);
        const uint32_t dtypes[]{RAD_I8, RAD_F32, RAD_I8, packed ? RAD_F32 : RAD_BF16, RAD_BF16};
        const int widths[]{reduction, groups, reduction, groups, columns};
        for (int operand = 0; operand < 7; ++operand) {
            RadOpdDesc desc{};
            REQUIRE_EQ(row->opd_shape(resolved.geom.params(), resolved.geom.n_params(), operand, &desc), RAD_OK);
            if (operand >= 5) { CHECK(desc.flags & RAD_OPD_F_ABSENT); continue; }
            CHECK(!(desc.flags & RAD_OPD_F_ABSENT));
            CHECK_EQ(desc.rank, 2u);
            CHECK_EQ(desc.dtype, dtypes[operand]);
            CHECK_EQ(desc.shape[0], operand == 2 || operand == 3 ? columns : band.hi);
            CHECK_EQ(desc.shape[1], widths[operand]);
        }
    }
    REQUIRE_EQ(builder.finish(), RAD_OK);
    const auto& codes = *builder.program().weight(weight);
    const auto& scales = *builder.program().weight(scale);
    CHECK_EQ(codes.n_sel, packed ? 3 : 1);
    CHECK_EQ(scales.n_sel, factored ? 2 : 1);
    CHECK_EQ(codes.identity, !packed);
    CHECK_EQ(scales.identity, !packed);
    CHECK_EQ(codes.stored_dtype, packed ? r3d_iqr::weight_dtype : uint32_t(RAD_I8));
    CHECK_EQ(scales.stored_dtype, !packed ? uint32_t(RAD_BF16) : factored ? r3d_iqr::scale_dtype : r3d_iqr::group_scale_dtype);
    CHECK_EQ(codes.stored_bytes, packed ? int64_t(sizeof(r3d_iqr::Header)) + rad_dtype_bytes(encoding.plane[0].dtype, columns * reduction) : columns * reduction);
    CHECK_EQ(scales.stored_bytes, factored ? (columns + groups) * 4 : columns * groups * 2);
    CHECK_EQ(scales.stored_shape[1], groups);
    CHECK_EQ(scales.decl.shape[1], factored ? reduction : groups);
}
}

TEST(r3d_iqr_registered_dense_declaration) {
    rad::Registry registry;
    REQUIRE_EQ(registry.load_plugin(R3D_IQR_FIXTURE, 0), RAD_OK);
    REQUIRE_EQ(registry.check_complete(), RAD_OK);
    declare(registry, rad_enc_plain(RAD_I8), false);
    declare(registry, rad_enc_affine(RAD_I8, RAD_BF16, 1, 128), false);
    for (auto dtype : {RAD_U3, RAD_U4, RAD_U5}) for (bool factored : {false, true}) {
        RadEncoding enc{};
        rad_enc_copy_str(enc.scheme, "trellis");
        rad_enc_copy_str(enc.transform, "fwht128");
        rad_enc_add_plane(&enc, "codes", dtype, 1, 1);
        rad_enc_add_plane(&enc, "scale", factored ? RAD_F32 : RAD_BF16, 1, factored ? 0 : 128);
        const int table = rad_enc_add_table(&enc, "trellis", RAD_I8, 65536, 1);
        enc.plane[table].block[0] = factored ? 256 : 128;
        enc.plane[table].block[1] = 1;
        rad_enc_add_table(&enc, "trellis.mad", RAD_U32, 2, 1);
        if (factored) rad_enc_add_plane(&enc, "scale.col", RAD_F32, 0, 128);
        declare(registry, enc, factored);
    }
}
RAD_TEST_MAIN()
