
include!("neuralnet.rs");
include!("activations.rs");
include!("eo_trainer.rs");
include!("seq_eo.rs");

extern crate eoalib;
use eoalib::*;

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{4,2,1};
    let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
    let mut nn = Neuralnet::new(layers, activations);
    
    //let trainer = Trainer {
     //   neuralnet : nn,
    //};   
    
    let mut wb = vec![0.0f64; nn.get_weights_biases_count()];

    for i in 0..wb.len() {
        wb[i] = i as f64;
    }    

    println!("W before update = {:?}", nn.weights);

    println!("biases before update = {:?}", nn.biases);

    nn.update_weights_biases(&wb);

    println!("W after update = {:?}", nn.weights);
    
    println!("biases after update = {:?}", nn.biases);

    let no : usize = 10;
    let dim : usize = 5;
    let kmax : usize =100;
    let lb : f64 =-10.0;
    let ub : f64 = 10.0;
    {
    let (a,b,c) = sequential_eo(no, kmax, lb, ub, dim, &fobj1);
    println!("best fitness = {:?}", a);
    println!("best solution = {:?}", b);
    println!("final best chart = {:?}", c[c.len()-1]);

    }    
}

fn fobj1(x : &Vec<f64>)-> f64 {
    let mut sum= 0.0f64;
    for i in 0..x.len(){
        sum += x[i];
    } 
    sum
}

   