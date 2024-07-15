use ndarray::Array2;
use ndarray_linalg::{Eig, Inverse};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cannot calculate the square root of a non-square matrix.")]
    NotSquare,
    #[error("Could not calculate eigen values: {0}")]
    Eig(ndarray_linalg::error::LinalgError),
    #[error("Failed to compute the inverse of eigenvectors matrix: {0}")]
    EigVecInv(ndarray_linalg::error::LinalgError),
}

type Result<T> = std::result::Result<T, Error>;

/// Returns the real part of the square root of a matrix.
/// Good references:
/// - [wiki](https://en.wikipedia.org/wiki/Square_root_of_a_matrix)
/// - [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html)
pub fn sqrtm_re(a: &Array2<f32>) -> Result<Array2<f32>> {
    if a.nrows() != a.ncols() {
        return Err(Error::NotSquare);
    }

    // Compute the eigenvalues and eigenvectors
    let (eigvals, eigvecs) = a.eig().map_err(Error::Eig)?;

    // Take square roots of the eigenvalues
    let sqrt_eigvals = Array2::from_diag(&eigvals.mapv(|v| v.sqrt()));

    let v_inv = eigvecs.inv().map_err(Error::EigVecInv)?;
    let sqrt_a = eigvecs.dot(&sqrt_eigvals).dot(&v_inv);

    let sqrt_a = sqrt_a.view().split_complex();

    Ok(sqrt_a.re.to_owned())
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use super::sqrtm_re;

    #[test]
    fn test() {
        let mat = Array2::from_shape_vec((2, 2), vec![33.0, 24.0, 48.0, 57.0]).unwrap();
        let s = sqrtm_re(&mat).unwrap();
        let target = Array2::from_shape_vec((2, 2), vec![5.0, 2.0, 4.0, 7.0]).unwrap();

        assert!(s.abs_diff_eq(&target, 1e-3));
    }
}
