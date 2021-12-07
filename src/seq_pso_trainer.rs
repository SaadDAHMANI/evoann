// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)

// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)
// https://github.com/SaadDAHMANI/equilibrium_optimizer
//----------------------------------------------------------------------------------
//extern crate rand;
//extern crate rayon;
//use rand::distributions::Uniform;
//use rand::distributions::Distribution;
//use rayon::prelude::*;

pub struct SequentialPSOTrainer<'a>{
     neuralnet : &'a mut Neuralnet,
     particles : usize,
     dimension : usize,
     max_iterations : usize,
     upper_bound : f64,
     lower_bound : f64,
     c1 : f64,
     c2 : f64,
     //final_leran_error : f64,
     learn_in : Vec<Vec<f64>>,
     expected_learn_out : Vec<Vec<f64>>,
     //learning_curve : Vec<f64>,
     //best_weights_biases : Vec<f64>,
}
impl<'a> SequentialPSOTrainer<'a> {
     pub fn new(neuralnet: &'a mut Neuralnet, learnin : Vec<Vec<f64>>, learnout : Vec<Vec<f64>>, particles : usize, max_iter : usize , lb : f64, ub : f64)-> SequentialPSOTrainer {
          let mut newtrainer = SequentialPSOTrainer {
                  neuralnet : neuralnet,
                  particles : particles,
                  dimension : 0,
                  max_iterations : max_iter,
                  upper_bound : ub,
                  lower_bound : lb,
                  learn_in : learnin,
                  expected_learn_out : learnout,
                  c1 : 2.0,
                  c2 : 2.0,
              };
     
              newtrainer.dimension = newtrainer.neuralnet.get_weights_biases_count();
              newtrainer
        }    

        
    pub fn compute_out_for(&mut self, inputs : &Vec<f64>)-> Vec<f64>{
          self.neuralnet.feed_forward(&inputs)
     }

    pub fn compute_out_for2(&mut self, inputs : &Vec<Vec<f64>>)->Vec<Vec<f64>> {
          let mut result = Vec::new();
          for data in inputs.iter(){
          result.push(self.neuralnet.feed_forward(&data));
           }
     result
    }

    pub fn learn(&mut self)->(f64, Vec<f64>, Vec<f64>) {

          let incount = self.learn_in.len();
          let outcount = self.expected_learn_out.len();

          if incount != outcount {
                panic!("Problem with learniong dataset size : count of learning input items must be equals (=) to count of learning output items.");
          }   

          //search the best [Wi, bi] for learning step
          let (final_err, best_wb, learning_curve) = self.run_seq_pso();
 
          //make the best [Wi, bi] solution as neuralnet weights & biases   
          self.neuralnet.update_weights_biases(&best_wb);

          (final_err, best_wb, learning_curve)
     }


    fn objectif_fn(&mut self, genome : &Vec<f64>)->f64 {        
           self.neuralnet.update_weights_biases(&genome);
           //println!("genome[0] : {:?}", genome[0]);      

           let learn_error : f64 = self.neuralnet.compute_learning_error_rmse(&self.learn_in, &self.expected_learn_out); 
           //println!("learn_error : {:?}", learn_error);       
           return learn_error;
      }

 pub fn run_seq_pso(&mut self) -> (f64, Vec<f64>, Vec<f64>){
     
      let dim = self.dimension;
      let ub=self.upper_bound;
      let lb =self.lower_bound;
      let max_iter = self.max_iterations;
      let nop = self.particles;
         
      let mut cgcurve = vec![0.0f64; max_iter];
     
     // Define the PSO's paramters
     let w_max : f64 = 0.9;
     let w_min : f64 = 0.2;
     let c1 :f64 = 2.0;
     let c2 :f64 = 2.0;
     let v_max = (ub - lb)* 0.2f64;
     let v_min  = -1.0*v_max;
 
     
     // PSO algorithm
     
     // Particles initialization
     let mut particles = initialize(nop, dim, lb , ub);
     
     // Velocities initialization 
     let mut v = vec![vec![0.0f64; dim]; nop];
     
     //let mut currentx = Solution::new(nop+1, dim);
 
     let mut pbest_x = Vec::new();
     //let mut pbest_0 = Vec::new();
 
     let mut gbest_x = Solution::new(dim+100, dim);
     //let mut gbest_0 = Vec::new();
 
     let mut rand1 = vec![0.0f64; dim];
     let mut rand2 = vec![0.0f64; dim];
     
     for j in 0..nop{
         pbest_x.push(Solution::new(j, dim));
     }
 
     // Main PSO loop 
     for t in 0..max_iter {
         
         //Objective function computation
         for k in 0..nop {
 
              // Evaluate search agent using objective function 
              particles[k].fitness = self.objectif_fn(&particles[k].c); 
              //copy_solution(&particles[k], &mut currentx);
              //copy_solution(&particles[k], &mut currentx);             
 
             //Update the pbest 
             if particles[k].fitness < pbest_x[k].fitness {
                copy_solution(&particles[k], &mut pbest_x[k]);                 
             } 
 
             //Update the gbest
             if particles[k].fitness < gbest_x.fitness {
                 copy_solution(&particles[k], &mut gbest_x);
             }
         }
 
         //Update the x and v
         let tf64 = t as f64;
         let max_iterf64 = max_iter as f64;
         let w = w_max - ((tf64*(w_max - w_min))/ max_iterf64);
         
         for k in 0..nop{
             randomize(&mut rand1);
             randomize(&mut rand2);
 
             for j in 0..dim{
                  v[k][j] = (w*v[k][j]) + (c1*rand1[j]*(pbest_x[k].c[j]-particles[k].c[j])) + (c2*rand2[j]*(gbest_x.c[j]-particles[k].c[j]));                 
              } 
              
              //let mut index1 = Vec::new();
              //let mut index2 = Vec::new();
 
              for j in 0..dim{
                  if v[k][j]>v_max {
                      //index1.push(j);
                      v[k][j]=v_max;
                  }
 
                  if v[k][j]<v_min{
                      //index2.push(j);
                      v[k][j] = v_min;
                  }
              }
 
              // Update particles positions 
              for j in 0..dim{
                 particles[k].c[j] = particles[k].c[j] + v[k][j]; 
              }           
 
              for j in 0..dim{
                 if particles[k].c[j] > ub {
                     particles[k].c[j] = ub;
                 } 
 
                 if particles[k].c[j] < lb {
                     particles[k].c[j] = lb;
                 } 
             } 
         }
 
         cgcurve[t]= gbest_x.fitness;  
 
     }
 
     return (gbest_x.fitness, gbest_x.c, cgcurve);
 
 }
     








}
