use ndarray::{Array1, Array2, Axis, Ix1, Ix2};
use ndarray_npy::ReadNpyExt;
use std::{fs::File, io::Read}; // needed for the trait

fn load_data() -> (
    (Array2<f64>, Array1<u8>),
    (Array2<f64>, Array1<u8>),
    (Array2<f64>, Array1<u8>),
) {
    let mut f = File::open("train_x.npy").unwrap();
    let train_x = Array2::<f64>::read_npy(&mut f).unwrap();

    let mut f = File::open("train_y.npy").unwrap();
    let train_y = Array1::<u8>::read_npy(&mut f).unwrap();

    let mut f = File::open("valid_x.npy").unwrap();
    let valid_x = Array2::<f64>::read_npy(&mut f).unwrap();

    let mut f = File::open("valid_y.npy").unwrap();
    let valid_y = Array1::<u8>::read_npy(&mut f).unwrap();

    let mut f = File::open("test_x.npy").unwrap();
    let test_x = Array2::<f64>::read_npy(&mut f).unwrap();

    let mut f = File::open("test_y.npy").unwrap();
    let test_y = Array1::<u8>::read_npy(&mut f).unwrap();

    ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))
}

fn vectorized_result(j: usize) -> Array2<f64> {
    let mut e = Array2::zeros((10, 1));
    e[(j, 0)] = 1.0;
    e
}

pub fn load_data_wrapper() -> (
    Vec<(Array2<f64>, Array2<f64>)>,
    Vec<(Array2<f64>, u8)>,
    Vec<(Array2<f64>, u8)>,
) {
    let ((tr_x, tr_y), (va_x, va_y), (te_x, te_y)) = load_data();
    let training_data = tr_x
        .axis_iter(Axis(0))
        .zip(tr_y.iter())
        .map(|(x, y)| {
            (
                x.to_owned().into_shape_with_order((784, 1)).unwrap(),
                vectorized_result(*y as usize),
            )
        })
        .collect();
    let validation_data = va_x
        .axis_iter(Axis(0))
        .zip(va_y.iter())
        .map(|(x, y)| (x.to_owned().into_shape_with_order((784, 1)).unwrap(), *y))
        .collect();
    let test_data = te_x
        .axis_iter(Axis(0))
        .zip(te_y.iter())
        .map(|(x, y)| (x.to_owned().into_shape_with_order((784, 1)).unwrap(), *y))
        .collect();
    (training_data, validation_data, test_data)
}
