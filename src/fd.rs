//! [Fréchet Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) calculation

use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, Trace};

use crate::sqrtm::sqrtm_re;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to calculate the square root of a matrix: {0}")]
    Sqrtm(#[from] crate::sqrtm::Error),
    #[error("Failed to calculate the trace of a matrix: {0}")]
    Trace(ndarray_linalg::error::LinalgError),
}

type Result<T> = std::result::Result<T, Error>;

/// Calculating the Fréchet Distance between two normal (Gaussian) distributions.
pub fn fd_normal_arr(
    mu_x: &[f32],
    mu_y: &[f32],
    sigma_x: &[Vec<f32>],
    sigma_y: &[Vec<f32>],
) -> Result<f32> {
    let n = mu_x.len();

    fd_normal_nd(
        mu_x.to_vec().into(),
        mu_y.to_vec().into(),
        Array2::from_shape_fn([n, n], |(i, j)| sigma_x[i][j]),
        Array2::from_shape_fn([n, n], |(i, j)| sigma_y[i][j]),
    )
}

/// Calculating the Fréchet Distance between two normal (Gaussian) distributions.
pub fn fd_normal_nd(
    mu_x: Array1<f32>,
    mu_y: Array1<f32>,
    sigma_x: Array2<f32>,
    sigma_y: Array2<f32>,
) -> Result<f32> {
    let m = (mu_x - mu_y).norm_l2();

    let s = &sigma_x.dot(&sigma_y);
    let s = sqrtm_re(&s.to_owned())?;

    let v = &sigma_x + &sigma_y - s * 2.0;
    let v_trace = v.trace().map_err(Error::Trace)?;
    Ok(m + v_trace)
}
