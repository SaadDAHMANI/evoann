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

    pub fn learn(&mut self) {
         self.learn_dataset_size = self.learn_in.len();
         if self.learn_dataset_size != self.expected_learn_out.len() {
             panic!("Proble with learniong dataset size (size_in must be = size_out).");
         }   

         let dim = self.neuralnet.get_weights_biases_count();

         let (final_learn_err, best_weightsbiases, learn_curve) = seq_eo(self.population_size, self.max_iterations, -0.5, 0.5, dim, &mut self, &mut self.objfn());
         self.final_leran_error = final_learn_err;

    }

    fn objfn(&mut self, genome : &Vec<f64>)->f64 {
        
        self.neuralnet.update_weights_biases(&genome);

        let learn_error : f64 = self.compute_learning_error();  
        
        return learn_error;
    }

    fn compute_learning_error(&mut self)->f64 {

        let mut totalerror : f64 = 0.0f64;
        let mut err : f64 =0.0f64;

        for i in 0..self.learn_dataset_size {

            let computed = self.neuralnet.feed_forward(&self.learn_in[i]);

            for j in 0..computed.len(){
               err = f64::powi(self.expected_learn_out[i][j]- computed[j], 2); 
            }
            totalerror +=err;
        }
        totalerror
    }
    
    
}


#[cfg(test)]
mod trainer_tests {
// use super::*;

}
