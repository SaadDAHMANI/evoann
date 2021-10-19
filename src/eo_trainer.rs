pub struct Trainer {
     neuralnet : Neuralnet,
     population_size : usize,
     max_iteration : usize,
     final_error : f64,
     learn_in : Vec<Vec<f64>>,
     expected_learn_out : Vec<Vec<f64>>,
}

impl Trainer {

    fn objective_function(&self, genome : &Vec<f64>)->f64 {
        
                
        return 0.0f64;
    }

     fn learn(&self){



    }
    
    
}


#[cfg(test)]
mod trainer_tests {
// use super::*;

}
