// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)
// https://github.com/SaadDAHMANI/equilibrium_optimizer
//----------------------------------------------------------------------------------

extern crate rand;
extern crate rayon;
use rand::distributions::Uniform;
use rand::distributions::Distribution;
use rayon::prelude::*;

//------------objective function---------------
pub fn fobj(x : &Vec<f64>)-> f64 {
     
     //let sum : f64 = x.iter().map(|s| s.powi(2)).sum();
      //for value in x.iter(){
     //sum += value.powi(2);
     //}  
    
     //std::thread::sleep(std::time::Duration::from_millis(50));

     return f1(&x) + f2(&x) + f1(&x) + f2(&x);
 }
 
//---------------------------------------------

pub fn parallel_eo(particles_no : usize, max_iter : usize, lb : f64, ub : f64, dim : usize, number_of_threads: Option<usize>) -> Result<(f64, Solution, Vec<f64>), rayon::ThreadPoolBuildError> {
 
     //---------------------PARALLEL PARAMS ---------------------------------------------------
     let default_nbr_threads : usize = 4; //rayon::current_num_threads();

     let nbr_of_threads = match number_of_threads {
         Some(thrds) => 
             {
              
                 if (thrds>0) & (thrds<= particles_no) {
                         thrds
                    }
                 else if thrds > particles_no {
                        particles_no
                    }
                 else {default_nbr_threads} 
             },
         None => default_nbr_threads,
    };
   
    match rayon::ThreadPoolBuilder::new().num_threads(nbr_of_threads).build_global() {
        Ok(()) => {
          
            // Performe ParaEO (Parallel Equilibrium Optimizer)
   
    //---------------------------------------------------------------------------------------------        
      // Initialize variables 
      //Ceq1=zeros(1,dim);   Ceq1_fit=inf; 
      //Ceq2=zeros(1,dim);   Ceq2_fit=inf; 
      //Ceq3=zeros(1,dim);   Ceq3_fit=inf; 
      //Ceq4=zeros(1,dim);   Ceq4_fit=inf;
     let mut ceq1 = Solution::new(dim+1, dim);

     let mut ceq2 =  Solution::new(dim+2, dim);
     
     let mut ceq3 =  Solution::new(dim+3, dim);
     
     let mut ceq4 =  Solution::new(dim+4, dim);
     
     let mut ceq_ave =  Solution::new(dim+5, dim);

     let mut convergence_curve = vec![0.0f64; max_iter]; 

     let mut ceq1_fit = f64::MAX;
    
    let mut ceq2_fit = f64::MAX;
    
    let mut ceq3_fit = f64::MAX;
    
    let mut ceq4_fit = f64::MAX;

     //C=initialization(Particles_no,dim,ub,lb);
     let mut cs =initialize(particles_no, dim, lb, ub);


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
    let mut fitness = vec![0.0f64; particles_no];
    let mut fit_old = vec![0.0f64; particles_no];
    let mut cs_old = cs.clone();// vec![vec![0.0f64; dim]; particles_no];
    let mut c_pool = vec![vec![0.0f64; dim]; 5];
    
    while iter < max_iter {
         // space bound
          for sln in cs.iter_mut() {
               for j in 0..dim {
                     if sln.c[j] < lb { sln.c[j] = lb;}
                     if sln.c[j] > ub { sln.c[j] = ub;}
               }
          }

          //--------------- PARALLE --------------------
           // compute fitness for agents in parallel mode
           // fitness(i)=fobj(C(i,:));

           cs.par_iter_mut().for_each(|s| {
               s.fitness = fobj(&s.c)
          });  
           //------------------------------------------- 
            // copy fitness to vector 
            for sln in cs.iter() {
                 fitness[sln.index] = sln.fitness;
            }
           //---------------SEQUENTIAL -----------------
           for sln in cs.iter() {
               // check fitness with best 
                if sln.fitness < ceq1_fit {
                     ceq1_fit= sln.fitness;
                     copy_solution(&sln, &mut ceq1);
                }
               else if (sln.fitness < ceq2_fit) & (sln.fitness > ceq1_fit) {
                    ceq2_fit= sln.fitness;
                    copy_solution(&sln, &mut ceq2);            
               }
               else if (sln.fitness < ceq3_fit) & (sln.fitness > ceq1_fit) & (sln.fitness > ceq2_fit)   {
                    ceq3_fit= sln.fitness;
                    copy_solution(&sln, &mut ceq3);
               }
               else if (sln.fitness < ceq4_fit)  & (sln.fitness > ceq1_fit) & (sln.fitness > ceq2_fit) & (sln.fitness > ceq3_fit) {
                    ceq4_fit= sln.fitness;
                    copy_solution(&sln, &mut ceq4);
                }
           }
           
            //-- Memory saving---
            if iter == 0 {
               copy_vector(&fitness, &mut fit_old);
               copy_population(&cs, &mut cs_old);
           }
       
           for i in 0..particles_no {
               if fit_old[i] < fitness[i] {
                   fitness[i]=fit_old[i];
                   copy_solution(&cs_old[i], &mut cs[i]);
               }
           }

               
           copy_population(&cs, &mut cs_old);
           copy_vector(&fitness, &mut fit_old);

            // compute averaged candidate Ceq_ave 
           for i in 0..dim {
               ceq_ave.c[i] = (ceq1.c[i] + ceq2.c[i] + ceq3.c[i] + ceq4.c[i])/4.0;
          }
           //Equilibrium pool
          for i in 0..dim {
          c_pool[0][i] = ceq1.c[i];
          c_pool[1][i] = ceq2.c[i];
          c_pool[2][i] = ceq3.c[i];
          c_pool[3][i] = ceq4.c[i];
          c_pool[4][i] = ceq_ave.c[i];
          }

           // comput t using Eq 09
          let tmpt = (iter / max_iter) as f64;
          let t : f64 = (1.0 - tmpt).powf(a2*tmpt);
        
          //----------- update solution in PARALLEL mode ---------------  
          // let chrono = Instant::now(); 
          
          cs.par_iter_mut().for_each(|mut s|           
               {
                    update_c(&mut s, &c_pool, a1, gp, v, t);
               }
          );

          // let duration = chrono.elapsed();
          // println!("par --> End computation in : {:?}", duration);
  
          // println!("{:?}", cs[0].to_string());
  
          convergence_curve[iter] = ceq1_fit;
          iter +=1;
     }
     //return results
     Ok((ceq1_fit, ceq1, convergence_curve))

    },

    Err(error) => Err(error),     
    }  

}

fn initialize(searchagents_no : usize, dim : usize, lb : f64, ub : f64)-> Vec<Solution>{
    let mut positions : Vec<Solution> = vec![];
    
    let intervall01 = Uniform::from(0.0f64..=1.0f64);
    let mut rng = rand::thread_rng();              
    
    for i in 0..searchagents_no {
           let mut sln = Solution::new(i, dim);  
           for  j in 0..dim {   
            //  positions[i][j]= intervall01.sample(&mut rng)*(ub-lb)+lb;                         
            sln.c[j]= intervall01.sample(&mut rng)*(ub-lb)+lb;   
          }
          positions.push(sln);
    }    
    
    positions
}

fn update_c(sln : &mut Solution, c_pool : &Vec<Vec<f64>>, a1:f64, gp :f64, v :f64, t:f64) {
    
     let dim = sln.c.len();

    let mut lambda =  vec![0.0f64; dim];
    let mut r =  vec![0.0f64; dim];
    let mut r1 =  vec![0.0f64; dim];
    let mut r2 =  vec![0.0f64; dim];
    let mut ceq = vec![0.0f64; dim];
    let mut f = vec![0.0f64; dim];

    //andomize in 0..1
     let between = Uniform::from(0.0..=1.0); // [0.0, 1.0]
     let interval = Uniform::from(0..c_pool.len()); // [0, 5]
     let mut rng = rand::thread_rng();
     
     for i in 0..dim {
         lambda[i]=between.sample(&mut rng);
         r[i] = between.sample(&mut rng);
         r1[i] =between.sample(&mut rng);
         r2[i] =between.sample(&mut rng);
     }

     let _index = interval.sample(&mut rng);
     copy_vector(&c_pool[_index], &mut ceq);

        // compute F using Eq(11) 
        for j in 0..dim {
            f[j]=a1*f64::signum(r[j]-0.5)*(f64::exp(-1.0*lambda[j]*t)-1.0); 
        }
        let mut _g0 : f64 = 0.0;
        let mut _g :f64 = 0.0;
        let mut _gcp : f64 = 0.0;

        for j in 0..dim {
            // Eq. 15
            if r2[j]>gp { _gcp =0.5*r1[j]; }
            else {_gcp =0.0f64;}
         
            // Eq. 14
            _g0 = _gcp*(ceq[j]-lambda[j]*sln.c[j]);
           
            // Eq 13
            _g =_g0*f[j];
           
           // Eq. 16
            sln.c[j] = ceq[j]+(sln.c[j]-ceq[j])*f[j] +  (_g/(lambda[j]*v))*(1.0-f[j]); 
           } 
}

fn copy_solution(source : &Solution, destination : &mut Solution){
     destination.index = source.index;
     destination.fitness = source.fitness;

     for i in 0..source.c.len() {
         destination.c[i]=source.c[i];
     }
 }

 fn copy_population(source : & Vec<Solution>, destination : &mut Vec<Solution>) {
    
     let ni = source.len();
     let nj = source[0].c.len();
 
     for i in 0..ni {
         for j in 0..nj {
             destination[i].c[j] =source[i].c[j];
         }
      }
 }
