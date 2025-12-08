// #[macro_export]
// macro_rules! testgen_matmul_partition_buffering {
//     ($kind: ident, $algorithm: ty, $precision: ty, $selection_builder: expr) => {

#[cfg(not(feature = "matmul_tests_partition_buffering"))]
pub mod no_partition_buffering {
    use super::*;
    use cubek_matmul::components::stage::PartitionBuffering;

    fn partition_buffering() -> PartitionBuffering {
        PartitionBuffering::Single
    }

    include!("../problem/layouts.rs");
}
// $crate::testgen_matmul_problem!(
//     $kind,
//     $algorithm,
//     $precision,
//     $selection_builder.partition_buffering(PartitionBuffering::Single)
// );

// #[cfg(feature = "matmul_tests_partition_buffering")]
// mod pb1 {
//     use super::*;

//     $crate::testgen_matmul_problem!(
//         $kind,
//         $algorithm,
//         $precision,
//         $selection_builder.partition_buffering(PartitionBuffering::Single)
//     );
// }

// #[cfg(feature = "matmul_tests_partition_buffering")]
// mod pb2 {
//     use super::*;

//     $crate::testgen_matmul_problem!(
//         $kind,
//         $algorithm,
//         $precision,
//         $selection_builder.partition_buffering(PartitionBuffering::Double)
//     );
// }
//     };
// }
