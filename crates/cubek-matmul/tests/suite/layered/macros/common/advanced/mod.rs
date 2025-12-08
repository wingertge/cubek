// pub mod hypercube;
// mod partition_buffering;
// mod specialized;
// mod swizzle;

// // #[macro_export]
// // macro_rules! testgen_matmul_advanced {
// //     ($kind: ident, $algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
// use cubek_matmul::components::{MatmulSelection, MatmulSelectionBuilder};

// mod _advanced {
//     use super::*;
//     use cubecl::prelude::*;
//     use cubek_matmul::components::TilingSchemeBuilder;

//     pub fn get_selection_builder(builder: TilingSchemeBuilder) -> MatmulSelectionBuilder {
//         let tiling_scheme = builder.build().unwrap();
//         let client = cubecl::TestRuntime::client(&Default::default());
//         let plane_dim = client.properties().hardware.plane_size_max;
//         MatmulSelection::builder(tiling_scheme, plane_dim)
//     }

//     include!("specialized.rs");
// }

// // $crate::testgen_matmul_specialized!(
// //     $kind,
// //     $algorithm,
// //     $precision,
// //     _advanced::get_selection_builder()
// // );
// //     };
// // }
