mod algorithm;
mod precision;
mod tiling_scheme;

mod matmul_plane_vecmat {
    use cubek_matmul::components::tile::io::Filled;
    type TMM = cubek_matmul::components::tile::plane_vec_mat_inner_product::PlaneVecMatInnerProduct<
        Filled,
    >;

    #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_vecmat"))]
    include!("algorithm.rs");
}
