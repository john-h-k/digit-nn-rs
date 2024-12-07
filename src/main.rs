use network::Network;

mod mnist;
mod network;

fn main() {
    let (training_data, validation_data, test_data) = mnist::load_data_wrapper();

    let mut net = Network::new(vec![784, 30, 10]);

    eprintln!("Training");
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0, Some(test_data));

    // net.save("model").expect("Failed to save!");
}
