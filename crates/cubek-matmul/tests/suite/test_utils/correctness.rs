use std::any::TypeId;
use std::fmt::Display;

use cubecl::TestRuntime;
use cubecl::prelude::Numeric;
use cubecl::{CubeElement, client::ComputeClient, prelude::Float, server};

use crate::suite::test_utils::cpu_reference::{CastInto, matmul_cpu_reference};
use cubek_matmul::components::MatmulProblem;

pub fn assert_result<
    EG: Float + CubeElement + Display + CastInto<ES>,
    ES: Numeric + CastInto<EA>,
    EA: Numeric + CastInto<EG>,
>(
    lhs: &[EG],
    rhs: &[EG],
    problem: &MatmulProblem,
    client: &ComputeClient<TestRuntime>,
    out: server::Handle,
    shape: &[usize],
    strides: &[usize],
) {
    let eps_global = epsilon_for_type::<EG>();
    let eps_stage = epsilon_for_type::<ES>();
    let eps_acc = epsilon_for_type::<EA>();

    // Empirically chosen for metal
    let safety_factor = 170.0;
    let epsilon = (eps_global.max(eps_stage).max(eps_acc)) * safety_factor;

    let expected = matmul_cpu_reference::<EG, ES, EA>(lhs, rhs, problem)
        .into_iter()
        .collect::<Vec<EG>>();

    if let Err(e) = assert_equals_approx::<EG>(client, out, shape, strides, &expected, epsilon) {
        panic!("{}", e);
    }
}

fn epsilon_for_type<T: 'static>() -> f32 {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        f32::EPSILON
    } else {
        half::f16::EPSILON.to_f32()
    }
}

/// Compares the content of a handle to a given slice of f32.
fn assert_equals_approx<F: Float + CubeElement + Display>(
    client: &ComputeClient<TestRuntime>,
    output: server::Handle,
    shape: &[usize],
    strides: &[usize],
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let env = std::env::var("CUBEK_TEST_MODE");

    let print_instead_of_compare = match env {
        Ok(val) => matches!(val.as_str(), "print"),
        Err(_) => false,
    };

    let actual =
        client.read_one_tensor(output.copy_descriptor(shape, strides, F::type_size() as usize));
    let actual = F::from_bytes(&actual);

    let epsilon = epsilon.max(F::EPSILON.to_f32().unwrap());

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

        // account for lower precision at higher values
        if print_instead_of_compare {
            println!("{:?}: {:?}, {:?}", i, a, e);
        } else {
            let actual_nan = f32::is_nan(a.to_f32().unwrap());
            let expected_nan = f32::is_nan(e.to_f32().unwrap());

            if actual_nan != expected_nan {
                if expected_nan {
                    return Err(format!("Expected NaN, got value={:?}", *a));
                } else {
                    return Err(format!("Expected value={:?}, got NaN", *e));
                }
            }

            let difference = f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap());

            if difference >= allowed_error {
                return Err(format!(
                    "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                    i, *a, *e, difference, epsilon
                ));
            }
        }
    }

    if print_instead_of_compare {
        Err("".to_string())
    } else {
        Ok(())
    }
}
