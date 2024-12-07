use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::Array2;
use network::Network;

mod mnist;
mod network;

fn load_png_as_array(path: &str) -> Array2<f64> {
    let img = image::open(path).expect("failed to open image").to_luma8();
    let (w, h) = img.dimensions();

    let (min, max) = img
        .pixels()
        .fold((255, 0), |(min, max), p| (min.min(p[0]), max.max(p[0])));
    let range = (max as f64 - min as f64).max(1.0);

    let normalized_data: Vec<u8> = img
        .pixels()
        .map(|p| (((p[0] as f64 - min as f64) / range) * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    let normalized = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w, h, normalized_data)
        .expect("image buffer size mismatch");

    let resized = image::imageops::resize(&normalized, 28, 28, FilterType::Lanczos3);

    let data: Vec<f64> = resized
        .pixels()
        .map(|p| {
            let val = p[0] as f64 / 255.0;
            let val = if val > 0.6 { 1.0 } else { val };
            // invert so background ~0 and digit ~1
            1.0 - val
        })
        .collect();

    DynamicImage::ImageLuma8(ImageBuffer::from_fn(28, 28, |x, y| {
        let idx = (y * 28 + x) as usize;
        let v = (data[idx] * 255.0).clamp(0.0, 255.0) as u8;
        Luma([v])
    }))
    .save("normalised.png")
    .expect("failed to save image");

    Array2::from_shape_vec((784, 1), data).expect("shape error")
}

fn main() {
    let data = load_png_as_array("evie7.png");

    let (training_data, validation_data, test_data) = mnist::load_data_wrapper();

    let mut net = Network::new(vec![784, 30, 10]);
    eprintln!("Training");
    net.stochastic_gradient_descent(training_data, 100, 10, 3.0, Some(test_data));

    // let mut net = Network::load("model", vec![784, 30, 10]).unwrap();

    let res = net.evaluate(&[data])[0];

    println!("Result: {res}");

    net.save("model").expect("Failed to save!");
}
