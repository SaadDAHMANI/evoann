// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)

pub struct Trainer {
     neuralnet : Neuralnet,
     population_size : usize,
     max_iterations : usize,
     final_leran_error : f64,
     learn_in : Vec<Vec<f64>>,
     expected_learn_out : Vec<Vec<f64>>,
     learn_dataset_size : usize,
}

impl Trainer {
    pub fn new( nnet : Neuralnet, pop_size : usize, max_iter: usize, learn_in : Vec<Vec<f64>>, exp_learn_out : Vec<Vec<f64>>)-> Trainer{
        Trainer {
            neuralnet : nnet,
            population_size : pop_size,
            max_iterations : max_iter,
            learn_in : learn_in,
            expected_learn_out : exp_learn_out,
            final_leran_error : 0.0f64,
            learn_dataset_size : 0,
        }
    }

    pub fn learn(&mut self) {
        // self.learn_dataset_size = self.learn_in.len();

         if self.learn_dataset_size != self.expected_learn_out.len() {
             panic!("Problem with learniong dataset size : count of learning input items must be equals (=) to count of learning output items.");
         }   

        let dim = self.neuralnet.get_weights_biases_count();

        //let optimizer = SequentialEO::new(&mut self, self.population_size, dim, self.max_iterations, -0.5f64, 0.5f64, self.objectif_function2);

        // let (final_learn_err, best_weightsbiases, learn_curve) = seq_eo(&self, self.population_size, self.max_iterations, -0.5, 0.5, dim, &mut self, self.objfn2);
        // self.final_leran_error = final_learn_err;

    }

    fn objectif_function(&mut self, genome : &Vec<f64>)->f64 {        
        self.neuralnet.update_weights_biases(&genome);
        let learn_error : f64 = self.neuralnet.compute_learning_error(&self.learn_in, &self.expected_learn_out);         
        return learn_error;
    }







}



#[cfg(test)]
mod trainer_tests {
// use super::*;

}
