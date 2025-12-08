// #[macro_export]
// macro_rules! testgen_matmul_launch {
//     (Normal, $algorithm: ty, $precision: ty,  $selection: expr, $problem: expr) => {
// use cubecl::prelude::*;
// use cubek_matmul::components::MatmulElems;
// use $crate::suite::TestEG;
// use $crate::suite::TestES;
// use $crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;

use super::*;
// use crate::suite::layered::matmul_test_launcher::test_matmul_algorithm;
use cubecl::Runtime;
use cubecl::TestRuntime;
use cubek_matmul::components::MatmulElems;
use cubek_matmul::components::MatmulSelection;

include!("../../matmul_test_launcher.rs");

#[test]
pub fn test() {
    let client = TestRuntime::client(&Default::default());

    //     pub fn get_selection_builder(builder: TilingSchemeBuilder) -> MatmulSelectionBuilder {
    //         let tiling_scheme = builder.build().unwrap();
    //         let client = cubecl::TestRuntime::client(&Default::default());
    //
    //         MatmulSelection::builder(tiling_scheme, plane_dim)
    //     }
    let tiling_scheme = stage(partition(builder())).build().unwrap();
    let plane_dim = client.properties().hardware.plane_size_max;
    let selection_builder = MatmulSelection::builder(tiling_scheme, plane_dim);
    let matmul_selection = selection_builder
        .shared_swizzle(swizzle())
        .hypercube_config(hypercube_selection(&tiling_scheme))
        .partition_buffering(partition_buffering())
        .load_specialization_config(specialization())
        .build();

    matmul_test_launcher::test_matmul_algorithm::<Algorithm>(
        client,
        problem(),
        matmul_selection,
        MatmulElems::new::<Precision>(),
    );
}
// };

// (Tma, $algorithm: ty, $precision: ty, $selection: expr, $problem: expr) => {
//     use cubecl::prelude::*;
//     use $crate::suite::layered::tma_test_launcher::test_tma_matmul_algorithm;

//     #[test]
//     pub fn test() {
//         let client = cubecl::TestRuntime::client(&Default::default());
//         test_tma_matmul_algorithm::<$algorithm>(
//             client,
//             $problem,
//             $selection,
//             MatmulElems::new::<TestEG>(),
//         );
//     }
//     };
// }
