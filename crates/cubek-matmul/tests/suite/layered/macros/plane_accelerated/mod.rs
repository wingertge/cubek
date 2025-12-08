mod matmul_plane_accelerated {
    use super::*;
    use cubek_matmul::components::tile::io::Filled;
    pub type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

    // #[cfg(all(featurerow_fpests_plane", not(feature = "matmul_tests_mma")))]
    include!("algorithm.rs");

    // #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
    // mod cmma {
    //     use super::*;
    //     type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<Filled>;

    //     include!("algorithm.rs");
    // }

    // #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_mma"))]
    // mod mma {
    //     type TMM = cubek_matmul::components::tile::mma::MmaMatmul;

    //     include!("algorithm.rs");
    // }
}
