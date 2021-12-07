
extern crate rand;
//extern crate rayon;
use rand::distributions::Uniform;
use rand::distributions::Distribution;
//use rayon::prelude::*;


pub fn initialize(searchagents_no : usize, dim : usize, lb : f64, ub : f64)-> Vec<Solution>{
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

pub fn copy_solution(source : &Solution, destination : &mut Solution){
    destination.index = source.index;
    destination.fitness = source.fitness;

    for i in 0..source.c.len() {
        destination.c[i]=source.c[i];
    }
}

pub fn randomize(randvect : &mut Vec<f64>) {    
    let between = Uniform::from(0.0..=1.0);
    let mut rng = rand::thread_rng();
            
    for i in 0..randvect.len() {
        randvect[i]=between.sample(&mut rng);
    }
}