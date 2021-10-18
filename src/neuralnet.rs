use rand::Rng;

pub struct Neuralnet {
	pub neurons:Vec<Vec<f32>>,
	pub weights:Vec<Vec<Vec<f32>>>,
	pub biases:Vec<Vec<f32>>,
	pub activations:Vec<Activations>,
	pub layers:Vec<usize>,
	pub learning_rate: f32,
	pub cost: f32
}

impl Neuralnet {

	pub fn new(layers:Vec<usize>, activations:Vec<Activations>) -> Neuralnet
	{
		let mut nn = Neuralnet {
			neurons: Vec::new(),
			weights: Vec::new(),
			biases: Vec::new(),
			activations,
			layers,			
			learning_rate: 0.01,
			cost: 0.0
		};

		nn.init_neurons();
		nn.init_biases();
		nn.init_weights();

		return nn;

	}

	pub fn init_neurons(&mut self)
	{
		self.neurons = Vec::new();
		for x in 0..self.layers.len() {
			let neurons = vec![0.0; self.layers[x]];
			self.neurons.push(neurons);
		}
	}

	pub fn init_biases(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.biases = Vec::with_capacity(self.layers.len());

		for i in 1..self.layers.len() {
			let num_items = self.layers[i];
			let mut bias : Vec<f32> = vec![0.0; num_items];
			for i in 0..num_items {
				bias[i] = rng.gen_range(-0.5..0.5) / num_items as f32;
			}
			self.biases.push(bias);
		}
	}

	pub fn init_weights(&mut self)
	{
		let mut rng = rand::thread_rng();
		self.weights = Vec::new();

		for i in 1..self.layers.len() {
			let num_prev_items = self.layers[i-1];
			let mut layer_weights : Vec<Vec<f32>> = Vec::new();
			for _j in 0..self.layers[i] {
				let mut neuron_weights : Vec<f32> = vec![0.0; num_prev_items];
				for k in 0..num_prev_items {
					neuron_weights[k] = rng.gen_range(-0.5..0.5) / num_prev_items as f32;
				}
				layer_weights.push(neuron_weights);
			}
			self.weights.push(layer_weights);
		}
	}

	pub fn activate(&self, x:f32, layer_id: usize) -> f32 {
		match self.activations[layer_id] {
			Activations::Sigmoid => sigmoid(x),
			Activations::ReLU => relu(x),
			Activations::LeakyRelu => leakyrelu(x),
			Activations::TanH => tanh(x),
			Activations::SoftMax => softmax(x),
			_ => x
		}
	}

	pub fn feed_forward(&mut self, inputs:&Vec<f32>) -> Vec<f32> {
		for i in 0..inputs.len() {
			self.neurons[0][i] = inputs[i];
		}
		for i in 1..inputs.len() {
			let layer_idx = i - 1;

			for j in 0..self.layers[i] {
				let mut value:f32 = 0.0;
				for k in 0..self.layers[i-1] {
					value += self.weights[i - 1][j][k] * self.neurons[i - 1][k];
				}
				self.neurons[i][j] = self.activate(value + self.biases[i - 1][j], layer_idx);
			}

			match self.activations[layer_idx] {
				Activations::SoftMax => {
					let mut sigma : f32 = 0.0;
					for j in 0..self.layers[i] {
						sigma += self.neurons[i][j];
					}
					for j in 0..self.layers[i] {
						self.neurons[i][j] /= sigma;
					}
				},
				_ => {}
			}
		}
		self.neurons[self.layers.len() - 1].clone()
	}

	




	

}
