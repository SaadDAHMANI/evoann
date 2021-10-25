
include!("neuralnet.rs");
include!("activations.rs");
//include!("eo_trainer.rs");
include!("seq_eo_trainer.rs");

//extern crate eoalib;
//se eoalib::*;

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{2,1};
    let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear };
    let mut nn = Neuralnet::new(layers, activations);
    
    let mut wb = vec![0.0f64; nn.get_weights_biases_count()];

    for i in 0..wb.len() {
       wb[i] = i as f64;
    }    

    println!("W before update = {:?}", nn.weights);

    println!("biases before update = {:?}", nn.biases);

    nn.update_weights_biases(&wb);

    println!("W after update = {:?}", nn.weights);
    
    println!("biases after update = {:?}", nn.biases);

    println!("---------------------------------------------");
      

      let n : usize = 50;
      let  data_in = getdata_in(n);
      let  data_out = getdata_out(&data_in);
      
      //println!("In : {:?}", data_in);
      //println!("Out : {:?}", data_out);

      let p_size : usize = 20;
      let k_max : usize = 500;
      let ub : f64 = 10.00;
      let lb : f64 = -10.00;

      let mut eoann = SequentialEOTrainer::new(nn, data_in, data_out,p_size, k_max, lb, ub);
      
      let (_a, _b, _c) = eoann.learn();

      println!("final learning error : {:?}", _a );
      //println!("b : {:?}", _b );
      //println!("c : {:?}", _c );   
      let mut test = vec![0.0f64; 2];
      test[0] = 0.21;
      test[1] = 0.21;

      println!("testing result {:?} --> {:?}", test, eoann.compute_out_for(&test));

  
}

fn getdata_in(n : usize)->Vec<Vec<f64>> {
    let d=2;
    let mut positions = vec![vec![0.0f64; d]; n];
    let intervall01 = Uniform::from(0.0f64..=1.0f64);
    let mut rng = rand::thread_rng();              
    
    for i in 0..n {
         for  j in 0..d {   
              positions[i][j]= intervall01.sample(&mut rng);                         
         }
    }        
    positions
}

fn getdata_out(data : &Vec<Vec<f64>>)->Vec<Vec<f64>> {
    let d= data[0].len();
    let n = data.len();
    let mut positions = vec![vec![0.0f64; 1]; n];
    for i in 0..n {
        for j in 0..d {
            positions[i][0] += data[i][j]/2.0;
        }
    }
    positions
}

