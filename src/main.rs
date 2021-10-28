
include!("neuralnet.rs");
include!("activations.rs");
//include!("eo_trainer.rs");
include!("seq_eo_trainer.rs");

//extern crate eoalib;
//se eoalib::*;
use std::error::Error;
use csv;

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{2,3,1};
    let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear};
    let mut nnet = Neuralnet::new(layers, activations);
    
    let mut wb = vec![0.0f64; nnet.get_weights_biases_count()];

    for i in 0..wb.len() {
       wb[i] = i as f64;
    }    

    //println!("W before update = {:?}", nn.weights);

    //println!("biases before update = {:?}", nn.biases);

    //nn.update_weights_biases(&wb);

    //println!("W after update = {:?}", nn.weights);
    
    //println!("biases after update = {:?}", nn.biases);

    println!("---------------------------------------------");
      

      let n : usize = 50;
      let  data_in = getdata_in(n);
      let  data_out = getdata_out(&data_in);
      
      //println!("In : {:?}", data_in);
      //println!("Out : {:?}", data_out);

      let p_size : usize = 10;
      let k_max : usize = 500;
      let ub : f64 = 1.00;
      let lb : f64 = -1.00;

      let mut eoann = SequentialEOTrainer::new(&mut nnet, data_in, data_out,p_size, k_max, lb, ub);
      
      let (_a, _wbi, _c) = eoann.learn();
      
      println!("_");
      
      println!("final learning error : {:?}", _a );
      
      println!("_");

      println!("Best [Wi, bi] : {:?}", _wbi );
      //println!("c : {:?}", _c );   
      
       {
        println!("---------------------TESTING-----------------------"); 
        let mut bestnnet = nnet.clone();
        bestnnet.update_weights_biases(&_wbi);  
        
       } 
      
      

      let mut test = vec![0.0f64; 2];
      test[0] = 0.2;
      test[1] = 0.3;
      println!("--------------------------------------------"); 
      println!("_");

      println!("Real [Wi] = {:?}", nnet.weights);

      println!("Real [bi] = {:?}", nnet.biases);

      println!("_");
      
      println!("testing result Cos(...) {:?} --> {:?}", test, nnet.feed_forward(&test));
    
     


      

      //let path = "/home/sd/Documents/AppDev/Rust/evoann/data/data.csv";

       // if let Err(e)= read_from_file(&path) {
       //     eprintln!("{:?}", e)
       // }


  
}

fn getdata_in(n : usize)->Vec<Vec<f64>> {
    let d=2;
    let mut positions = vec![vec![0.0f64; d]; n];
    let intervall01 = Uniform::from(0.0f64..0.9f64);
    let mut rng = rand::thread_rng();              
    let mut value = 0.0f64;
    for i in 0..n {
         for  j in 0..d {   
            value =f64::round(intervall01.sample(&mut rng)*100.0);                         
           
            positions[i][j]= value/100.0;
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
            positions[i][0] += data[i][j];            
        }
        positions[i][0]= f64::cos(positions[i][0]);
        //println!("cos = {}",positions[i][0]);
    }
    positions
}

fn read_from_file(path : &str)-> Result<(), Box<dyn Error>> {
    
    let mut reader = csv::Reader::from_path(path)?;
    
    let headers = reader.headers()?;
    
    println!("Headers :  {:?}", headers);

    for result in reader.records() {
        let record = result?;

        println!("{:?}", record);
    }
    Ok(())
}  

