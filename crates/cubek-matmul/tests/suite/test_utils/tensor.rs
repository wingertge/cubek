use cubek_matmul::components::{MatmulIdent, MatmulProblem, MatrixLayout};

pub(crate) fn transpose<E: Copy>(array: &[E], batches: usize, rows: usize, cols: usize) -> Vec<E> {
    let mut result = vec![array[0]; array.len()];
    for b in 0..batches {
        for i in 0..rows {
            for j in 0..cols {
                result[(b * rows * cols) + j * rows + i] = array[(b * rows * cols) + i * cols + j];
            }
        }
    }
    result
}

/// Returns the total number of elements for the identified tensor, inferred by the problem definition
pub(crate) fn tensor_size(problem: &MatmulProblem, ident: MatmulIdent) -> usize {
    match ident {
        MatmulIdent::Lhs => problem.num_batches() * problem.m * problem.k,
        MatmulIdent::Rhs => problem.num_batches() * problem.k * problem.n,
        MatmulIdent::Out => problem.num_batches() * problem.m * problem.n,
    }
}

/// Returns the stride of the identified tensor, inferred by the problem definition
pub(crate) fn strides(problem: &MatmulProblem, ident: MatmulIdent) -> Vec<usize> {
    let shape = problem.shape(ident);
    let rank = shape.len();
    let mut strides = Vec::with_capacity(rank);

    let (last_batch, x, y) = match ident {
        MatmulIdent::Lhs => match problem.lhs_layout {
            MatrixLayout::RowMajor => (problem.m * problem.k, problem.k, 1),
            MatrixLayout::ColMajor => (problem.m * problem.k, 1, problem.m),
        },
        MatmulIdent::Rhs => match problem.rhs_layout {
            MatrixLayout::RowMajor => (problem.k * problem.n, problem.n, 1),
            MatrixLayout::ColMajor => (problem.k * problem.n, 1, problem.k),
        },
        MatmulIdent::Out => (problem.m * problem.n, problem.n, 1),
    };

    strides.push(y);
    strides.push(x);

    if rank > 2 {
        strides.push(last_batch);

        for b in shape.iter().rev().take(rank - 3) {
            strides.push(last_batch * b)
        }
    }

    strides.into_iter().rev().collect()
}
