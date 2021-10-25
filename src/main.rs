
include!("neuralnet.rs");
include!("activations.rs");
//include!("eo_trainer.rs");
include!("seq_eo_trainer.rs");

extern crate eoalib;
use eoalib::*;

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{2,2,1};
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

      let mut v = vec![0.0f32; 2];
      let x : f32 =10.0;
      let r = call_function(&mut v, add_one, x);
      println!("r= {:?}, v ={:?}", r, v);
    
      let mut data_in : Vec<Vec<f64>> = vec![vec![0.0f64; 2]; 10];
      let mut data_out : Vec< Vec<f64>> = vec![vec![0.0f64; 1]; 10];
      
      for i in 0..10 {
        data_in[i] = getdata_in(i, i+1);
        data_out[i] = getdata_out(i);
      }
      
      println!("In : {:?}", data_in);
      println!("Out : {:?}", data_out);

      let p_size : usize = 5;
      let k_max : usize = 2;
      let ub : f64 = 0.50;
      let lb : f64 = -0.50;

      let mut eoann = SequentialEOTrainer::new(nn, data_in, data_out,p_size, k_max, lb, ub);
      
      let (_a, _b, _c) = eoann.learn();

      println!("a : {:?}", _a );
      //println!("b : {:?}", _b );
      //println!("c : {:?}", _c );   
      let mut test = vec![0.0f64; 2];
      test[0] = 0.6;
      test[1] = 0.7;

      println!("testing {:?} --> {:?}", test, eoann.compute_out_for(&test));

  
}

fn getdata_in(i : usize, j : usize)->Vec<f64> {
    let mut v = vec![0.0f64;2];
    v[0]= i as f64;
    v[1]= j as f64;
    
    v[0]= v[0]/10.0;
    v[1]= v[1]/10.0;
    
    return v;
}

fn getdata_out(i : usize)->Vec<f64> {
    let mut v = vec![0.0f64;1];
    v[0]= i as f64;
       
    v[0]= v[0]/10.0;
    
    return v;
}


fn fobj1(x : &Vec<f64>)-> f64 {
    let mut sum= 0.0f64;
    for i in 0..x.len(){
        sum += x[i];
    } 
    sum
}


fn add_one(v : &mut Vec<f32>, x : f32)->f32 {
    v.push(x);
   return  x+1.0;
}

fn call_function( v: &mut Vec<f32>, f : fn(&mut Vec<f32>, f32)->f32, arg : f32)-> f32{

      let r = f(v, arg);
    println!("vect after : {:?}", v);
    r
}
   