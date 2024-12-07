use std::{
    fs::File,
    iter::{self, zip},
};

use ndarray_npy::{NpzReader, NpzWriter};
use rand::{seq::SliceRandom, RngCore};
use rand_distr::{Distribution, Normal};

use ndarray::{linalg::Dot, Array1, Array2, ArrayD, Axis};

pub trait NetworkFn {
    fn f(z: f64) -> f64;

    fn f_ref(z: &f64) -> f64 {
        Self::f(*z)
    }
}

struct Sigmoid;
impl NetworkFn for Sigmoid {
    fn f(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }
}

struct SigmoidPrime;
impl NetworkFn for SigmoidPrime {
    fn f(z: f64) -> f64 {
        Sigmoid::f(z) * (1.0 - Sigmoid::f(z))
    }
}

pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

type Vector = Array2<f64>;

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let biases: Vec<Array2<f64>> = sizes[1..]
            .iter()
            .map(|&y| Array2::from_shape_fn((y, 1), |_| dist.sample(&mut rng)))
            .collect();

        let weights: Vec<Array2<f64>> = sizes[..sizes.len() - 1]
            .iter()
            .zip(&sizes[1..])
            .map(|(&x, &y)| Array2::from_shape_fn((y, x), |_| dist.sample(&mut rng)))
            .collect();

        Self {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights,
        }
    }

    pub fn feed_forward(&mut self, mut a: Vector) -> Vector {
        for (b, w) in zip(&self.biases, &self.weights) {
            let d = w.dot(&a) + b;
            a = d.map(Sigmoid::f_ref);
        }

        a
    }

    pub fn stochastic_gradient_descent(
        &mut self,
        mut training_data: Vec<(Vector, Vector)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<Vec<(Vector, u8)>>,
    ) {
        let mut rng = rand::thread_rng();

        let n = training_data.len();

        for j in 0..epochs {
            training_data.shuffle(&mut rng);

            let mini_batches = training_data.chunks(mini_batch_size);
            let count = mini_batches.len();

            for (i, mini_batch) in mini_batches.enumerate() {
                // eprintln!("Doing batch {}/{}", i + 1, count);
                self.update_mini_batch(mini_batch, eta as _);
            }

            if let Some(ref test_data) = test_data {
                eprintln!("Evaluating epoch {j}...");
                let result = self.evaluate_test(test_data);
                let len = test_data.len();
                eprintln!("Epoch {j}: {result} / {len}");
            } else {
                eprintln!("Epoch {j} complete");
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Vector, Vector)], eta: usize) {
        let mut nabla_b = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.raw_dim()))
            .collect::<Vec<_>>();

        let mut nabla_w = self
            .weights
            .iter()
            .map(|b| Array2::<f64>::zeros(b.raw_dim()))
            .collect::<Vec<_>>();

        for (x, y) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);

            nabla_b = zip(nabla_b, delta_nabla_b)
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = zip(nabla_w, delta_nabla_w)
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }

        self.weights = zip(&self.weights, &nabla_w)
            .map(|(w, nw)| w - (eta as f64 / mini_batch.len() as f64) * nw)
            .collect();

        self.biases = zip(&self.biases, &nabla_b)
            .map(|(b, nb)| b - (eta as f64 / mini_batch.len() as f64) * nb)
            .collect();
    }

    fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        // init nabla_b and nabla_w with zeros matching shape of biases/weights
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.raw_dim()))
            .collect();

        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.raw_dim()))
            .collect();

        // feedforward
        let mut activation = x.clone();
        let mut activations = vec![x.clone()];
        let mut zs = vec![];

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            // z = w.dot(activation) + b
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = z.map(Sigmoid::f_ref);
            activations.push(activation.clone());
        }

        let delta = self.cost_derivative(&activations[activations.len() - 1], y)
            * zs[zs.len() - 1].map(SigmoidPrime::f_ref);

        *nabla_b.last_mut().unwrap() = delta.clone();

        let delta_col = delta; // delta already (y,1)
        let last_activation = &activations[activations.len() - 2]; // (x,1)
        let nabla_w_last = delta_col.dot(&last_activation.t());
        *nabla_w.last_mut().unwrap() = nabla_w_last;

        // for l in 2..self.num_layers
        for l in 2..=self.num_layers - 1 {
            let z = &zs[zs.len() - l];
            let sp = z.map(SigmoidPrime::f_ref);
            let w_next = &self.weights[self.weights.len() - l + 1];
            let delta_new = w_next.t().dot(&delta_col) * sp;
            let delta_col = delta_new;
            nabla_b[self.biases.len() - l] = delta_col.clone();
            let activation_prev = &activations[activations.len() - l - 1];
            nabla_w[self.weights.len() - l] = delta_col.dot(&activation_prev.t());
        }

        (nabla_b, nabla_w)
    }

    fn cost_derivative(&self, output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        output_activations - y
    }

    pub fn evaluate(&mut self, test_data: &[Array2<f64>]) -> Vec<usize> {
        test_data
            .iter()
            .map(|x| {
                let output = self.feed_forward(x.clone());

                let flat = output
                    .clone()
                    .into_shape_with_order((output.len(),))
                    .unwrap();

                let (argmax_idx, _) = flat
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                argmax_idx
            })
            .collect()
    }

    pub fn evaluate_test(&mut self, test_data: &[(Array2<f64>, u8)]) -> usize {
        test_data
            .iter()
            .map(|(x, y)| {
                let output = self.feed_forward(x.clone());

                let flat = output
                    .clone()
                    .into_shape_with_order((output.len(),))
                    .unwrap();
                let (argmax_idx, _) = flat
                    .indexed_iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                if argmax_idx == *y as usize {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut npz = NpzWriter::new(File::create(path)?);
        for (i, w) in self.weights.iter().enumerate() {
            npz.add_array(&format!("weight_{}", i), w).unwrap();
        }
        for (i, b) in self.biases.iter().enumerate() {
            npz.add_array(&format!("bias_{}", i), b).unwrap();
        }
        Ok(())
    }

    pub fn load(path: &str, sizes: Vec<usize>) -> std::io::Result<Self> {
        let mut npz = NpzReader::new(File::open(path)?).unwrap();
        let num_layers = sizes.len();
        let mut biases = Vec::new();
        let mut weights = Vec::new();

        for i in 0..num_layers - 1 {
            let w: Array2<f64> = npz.by_name(&format!("weight_{}", i)).unwrap();
            weights.push(w);
        }

        for i in 0..num_layers - 1 {
            let b: Array2<f64> = npz.by_name(&format!("bias_{}", i)).unwrap();
            biases.push(b);
        }

        Ok(Self {
            num_layers,
            sizes,
            biases,
            weights,
        })
    }
}
