use crate::fd::fd_normal_nd;
use crate::fid_forward::FidForward;
use burn::prelude::*;
use ndarray::{Array1, Array2, Axis};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("FD error")]
    Trace(#[from] crate::fd::Error),
}

type Result<T> = std::result::Result<T, Error>;

/// Fréchet inception distance
///
/// Good reference:
/// - [wiki](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
pub trait Fid<B: Backend>: FidForward<B> {
    fn fid(&self, a: Tensor<B, 4>, b: Tensor<B, 4>, layer: usize) -> Result<f32> {
        let a = tensor_to_nd(self.fid_forward(a, layer));
        let b = tensor_to_nd(self.fid_forward(b, layer));

        let (mu_a, sigma_a) = fit_gaussian(a);
        let (mu_b, sigma_b) = fit_gaussian(b);

        let out = fd_normal_nd(mu_a, mu_b, sigma_a, sigma_b)?;
        Ok(out)
    }
}

fn tensor_to_nd<B: Backend>(x: Tensor<B, 4>) -> Array2<f32> {
    let x = x.flatten::<2>(1, 3).into_data();
    let shape = [x.shape[0], x.shape[1]];
    let x = x.convert::<f32>();
    let v = x.as_slice::<f32>().unwrap().to_vec();
    Array2::from_shape_vec(shape, v).unwrap()
}

pub fn fit_gaussian(data: Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let batch = data.shape()[0];
    let n = data.shape()[1];
    let mean = data.mean_axis(Axis(0)).unwrap();

    let mut cov = Array2::<f32>::zeros([n, n]);
    for i in 0..n {
        for j in 0..=i {
            let mut total = 0.0;
            for k in 0..batch {
                let x = data[[k, i]] - mean[i];
                let y = data[[k, j]] - mean[j];
                total += x * y;
            }
            let value = total / batch as f32;
            cov[[i, j]] = value;
            cov[[j, i]] = value;
        }
    }

    (mean, cov)
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, Array2};

    use crate::fid::fit_gaussian;

    #[test]
    pub fn test_gaussian() {
        let data =
            Array2::from_shape_vec([3, 3], vec![0.0, 2.0, 0.0, 3.0, 4.0, 3.0, 5.0, 10.0, 1.0])
                .unwrap();

        let (mean, cov) = fit_gaussian(data);

        let target_mean =
            Array1::from_shape_vec([3], vec![2.66666667, 5.33333333, 1.33333333]).unwrap();

        let target_cov = Array2::from_shape_vec(
            [3, 3],
            vec![
                4.22222222,
                6.44444444,
                1.11111111,
                6.44444444,
                11.55555556,
                0.22222222,
                1.11111111,
                0.22222222,
                1.55555556,
            ],
        )
        .unwrap();

        mean.abs_diff_eq(&target_mean, 1e-3);
        cov.abs_diff_eq(&target_cov, 1e-3);
    }
}
