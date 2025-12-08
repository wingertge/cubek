// #[cfg(feature = "matmul_tests_f16")]
mod f16_ty {
    use super::*;
    type Precision = half::f16;
    pub type TestEG = Precision;
    pub type TestES = Precision;
    pub type TestEA = Precision;

    include!("tiling_scheme/tile.rs");

    // use super::*;

    // crate::testgen_matmul_accelerated_tiling_scheme!($algorithm, half::f16);
}

// #[cfg(feature = "matmul_tests_f32")]
// mod f32_ty {
//     use super::*;

//     crate::testgen_matmul_accelerated_tiling_scheme!($algorithm, f32);
// }
