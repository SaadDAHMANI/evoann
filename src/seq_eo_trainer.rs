// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)
// https://github.com/SaadDAHMANI/equilibrium_optimizer
//----------------------------------------------------------------------------------
extern crate rand;
//extern crate rayon;
use rand::distributions::Uniform;
use rand::distributions::Distribution;
//use rayon::prelude::*;

pub struct SequentialEOTrainer {
     neuralnet : Neuralnet,
     particles : usize,
     dimension : usize,
     max_iterations : usize,
     upper_bound : f64,
     lower_bound : f64,
     //final_leran_error : f64,
     learn_in : Vec<Vec<f64>>,
     expected_learn_out : Vec<Vec<f64>>,
     //learning_curve : Vec<f64>,
     //best_weights_biases : Vec<f64>,
}

impl SequentialEOTrainer {

    pub fn new(neuralnet: Neuralnet, learnin : Vec<Vec<f64>>, learnout : Vec<Vec<f64>>, particles : usize, max_iter : usize , lb : f64, ub : f64)-> SequentialEOTrainer {
        SequentialEOTrainer {
             neuralnet : neuralnet,
             particles : particles,
             dimension : 0,
             max_iterations : max_iter,
             upper_bound : ub,
             lower_bound : lb,
             learn_in : learnin,
             expected_learn_out : learnout,
         } 
   }    

   pub fn compute_out_for(&mut self, inputs : &Vec<f64>)-> Vec<f64>{
    self.neuralnet.feed_forward(&inputs)
}


    pub fn learn(&mut self)->(f64, Vec<f64>, Vec<f64>) {

        let incount = self.learn_in.len();
        let outcount = self.expected_learn_out.len();

        if incount != outcount {
         panic!("Problem with learniong dataset size : count of learning input items must be equals (=) to count of learning output items.");
     }   

     self.dimension = self.neuralnet.get_weights_biases_count();

     let (a, b, c) = self.run_seq_eo();

     (a,b,c)

     //self.best_weights_biases = b;
     //self.learning_curve = c;
}

    fn objectif_fn(&mut self, genome : &Vec<f64>)->f64 {        
        self.neuralnet.update_weights_biases(&genome);

        //println!("genome : {:?}", genome);      

        let learn_error : f64 = self.neuralnet.compute_learning_error(&self.learn_in, &self.expected_learn_out); 
        //println!("learn_error : {:?}", learn_error);       
        return learn_error;
}

pub fn run_seq_eo(&mut self) -> (f64, Vec<f64>, Vec<f64>) {
    
    let dim = self.dimension;
    let ub=self.upper_bound;
    let lb =self.lower_bound;

    // Initialize variables 
    //Ceq1=zeros(1,dim);   Ceq1_fit=inf; 
    //Ceq2=zeros(1,dim);   Ceq2_fit=inf; 
    //Ceq3=zeros(1,dim);   Ceq3_fit=inf; 
    //Ceq4=zeros(1,dim);   Ceq4_fit=inf;
    
    let mut ceq1 = vec![0.0f64; dim];
    
    let mut ceq2 = vec![0.0f64; dim];
    
    let mut ceq3 = vec![0.0f64; dim];
    
    let mut ceq4 = vec![0.0f64; dim];
    
    let mut ceq_ave = vec![0.0f64; dim];
    
    let mut ceq1_fit = f64::MAX;
    
    let mut ceq2_fit = f64::MAX;
    
    let mut ceq3_fit = f64::MAX;
    
    let mut ceq4_fit = f64::MAX;
    
    
    //C=initialization(Particles_no,dim,ub,lb);
    let mut c = self.initialization();
    
    // Iter=0; V=1;
    let mut iter =0;
    let v : f64 = 1.0;
    
    //
    // a1=2;
    // a2=1;
    // GP=0.5;
    let a1 : f64 = 2.0;
    let a2 : f64 = 1.0;
    let gp : f64 = 0.5;
    
    // to store agents fitness values
    let mut fitness = vec![0.0f64; self.particles];
    let mut fit_old = vec![0.0f64; self.particles];
    let mut c_old = vec![vec![0.0f64; dim]; self.particles];
    let mut c_pool = vec![vec![0.0f64; dim]; 5];
    let mut lambda = vec![0.0f64; dim];
    let mut r = vec![0.0f64; dim];
    let mut r1 = vec![0.0f64; dim];
    let mut r2 = vec![0.0f64; dim];
    let mut ceq = vec![0.0f64; dim];
    let mut f = vec![0.0f64; dim];
    let mut _gcp :f64 =0.0;
    //------------------------------------------
     let interval = Uniform::from(0..c_pool.len());
     //let between01 = Uniform::from(0.0..=1.0);
     let mut rng = rand::thread_rng();
    //------------------------------------------
    
    let mut convergence_curve = vec![0.0f64; self.max_iterations]; 
    let mut _index : usize = 0;
    let mut _g0 : f64 = 0.0; 
    let mut _g : f64 = 0.0;    
    
    while iter < self.max_iterations{
    
        for i in 0..c.len() {
    
            // space bound
             for j in 0..dim {
                if c[i][j] < lb { c[i][j] = lb;}
               if c[i][j] > ub { c[i][j] = ub;}
            }
    
            // compute fitness for agents
            
            //fitness[i] = fobj(&mut self.target_object, &c[i]);
            fitness[i] = self.objectif_fn(&c[i]);
            

            // check fitness with best 
            if fitness[i] < ceq1_fit {
                ceq1_fit= fitness[i];
                self.copy_vector(&c[i], &mut ceq1);
            }
            else if (fitness[i] < ceq2_fit) & (fitness[i] > ceq1_fit) {
                ceq2_fit= fitness[i];
                self.copy_vector(&c[i], &mut ceq2);            
            }
            else if (fitness[i] < ceq3_fit) & (fitness[i] > ceq2_fit) & (fitness[i] > ceq1_fit) {
                ceq3_fit= fitness[i];
                self.copy_vector(&c[i], &mut ceq3);
            }
            else if (fitness[i] < ceq4_fit) & (fitness[i] > ceq3_fit) & (fitness[i] > ceq2_fit) & (fitness[i] > ceq1_fit) {
                ceq4_fit= fitness[i];
                self.copy_vector(&c[i], &mut ceq4);
            }
        }
    
        //-- Memory saving---
    
        if iter == 0 {
            self.copy_vector(&fitness, &mut fit_old);
            self.copy_matrix(&c, &mut c_old);
        }
    
        for i in 0..self.particles {
            if fit_old[i] < fitness[i] {
                fitness[i]=fit_old[i];
                self.copy_vector(&c_old[i], &mut c[i]);
            }
        }
    
       self.copy_matrix(&c, &mut c_old);
       self.copy_vector(&fitness, &mut fit_old);
    
        // compute averaged candidate Ceq_ave 
        for i in 0..dim {
            ceq_ave[i] = (ceq1[i] + ceq2[i] + ceq3[i] + ceq4[i])/4.0;    
        }
    
        //Equilibrium pool
        for i in 0..dim {
            c_pool[0][i] = ceq1[i];
            c_pool[1][i] = ceq2[i];
            c_pool[2][i] = ceq3[i];
            c_pool[3][i] = ceq4[i];
            c_pool[4][i] = ceq_ave[i];
        }
    
        // comput t using Eq 09
        let tmpt = (iter / self.max_iterations) as f64;
        let t : f64 = (1.0 - tmpt).powf(a2*tmpt);

        // let chronos = Instant::now();
        
        for i in 0..self.particles {
    
            self.randomize(&mut lambda);        //  lambda=rand(1,dim);  lambda in Eq(11)
            self.randomize(&mut r);             //  r=rand(1,dim);  r in Eq(11  
                    
            //-------------------------------------------------------
            // Ceq=C_pool(randi(size(C_pool,1)),:); 
            // random selection of one candidate from the pool
             _index = interval.sample(&mut rng);
             self.copy_vector(&c_pool[_index], &mut ceq);
             //--------------------------------------------------------
             // compute F using Eq(11) 
             for j in 0..dim {
             f[j]=a1*f64::signum(r[j]-0.5)*(f64::exp(-1.0*lambda[j]*t)-1.0); 
         }
    
         // r1 and r2 to use them in Eq(15)
         self.randomize(&mut r1);
         self.randomize(&mut r2);
    
         for j in 0..dim {
             // Eq. 15
             if r2[j]>gp { _gcp =0.5*r1[j]; }
             else {_gcp =0.0f64;}
          
             // Eq. 14
             _g0 = _gcp*(ceq[j]-lambda[j]*c[i][j]);
            
             // Eq 13
             _g =_g0*f[j];
            
            // Eq. 16
             c[i][j] = ceq[j]+(c[i][j]-ceq[j])*f[j] +  (_g/(lambda[j]*v))*(1.0-f[j]); 
            }    
        }

        // let duration = chronos.elapsed();
        // println!("seq--> End computation in : {:?}", duration);
       
        convergence_curve[iter] = ceq1_fit;
        iter+=1;    
    }
    
    //return results
    (ceq1_fit, ceq1, convergence_curve)
    
    }
    
fn initialization(&self)-> Vec<Vec<f64>>{
        let mut positions = vec![vec![0.0f64; self.dimension]; self.particles];
        let intervall01 = Uniform::from(0.0f64..=1.0f64);
        let mut rng = rand::thread_rng();              
        
        for i in 0..self.particles {
             for  j in 0..self.dimension {   
                  positions[i][j]= intervall01.sample(&mut rng)*(self.upper_bound-self.lower_bound)+self.lower_bound;                         
             }
        }    
        
        positions
}
  

fn copy_matrix(&self, source : & Vec<Vec<f64>>, destination : &mut Vec<Vec<f64>>) {
    
        let ni = source.len();
        let nj = source[0].len();
    
        for i in 0..ni {
            for j in 0..nj {
                destination[i][j] =source[i][j];
            }
         }
}
   
fn randomize(&self, randvect : &mut Vec<f64>) {    
        let between = Uniform::from(0.0..=1.0);
        let mut rng = rand::thread_rng();
                
        for i in 0..randvect.len() {
            randvect[i]=between.sample(&mut rng);
        }
}
    
fn copy_vector(&self, source : & Vec<f64>, destination : &mut Vec<f64>){
    for i in 0..source.len() {
        destination[i]=source[i];
        }
 }
}
