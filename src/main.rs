
include!("neuralnet.rs");
include!("activations.rs");
//include!("eo_trainer.rs");
include!("seq_eo_trainer.rs");
include!("dataset.rs");

//extern crate eoalib;
//se eoalib::*;
use std::error::Error;

fn main() {
    println!("Hello, Evo-ANN!");

    //let layers:Vec<usize> = vec!{2,1,1};
    //let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear};
    //let mut nnet = Neuralnet::new(layers, activations);
    
    //let mut wb = vec![0.0f64; nnet.get_weights_biases_count()];

    //for i in 0..wb.len() {
    //   wb[i] = i as f64;
    //}    

    //println!("W before update = {:?}", nn.weights);

    //println!("biases before update = {:?}", nn.biases);

    //nn.update_weights_biases(&wb);

    //println!("W after update = {:?}", nn.weights);
    
    //println!("biases after update = {:?}", nn.biases);

    println!("---------------------------------------------");
      

      //let n : usize = 10;
      //let  data_in = getdata_in(n);
      //let  data_out = getdata_out(&data_in);
      
      //println!("In : {:?}", data_in);
      //println!("Out : {:?}", data_out);
      
      //let p_size : usize = 5;
      //let k_max : usize = 1;
      //let ub : f64 = 5.0;
      //let lb : f64 = -5.0;

      //let mut eoann = SequentialEOTrainer::new(&mut nnet, data_in, data_out,p_size, k_max, lb, ub);
      
      //let (_a, _wbi, _c) = eoann.learn();
      
      //println!("_");
      
      //println!("final learning error : RMSE = {:?}", _a );
      
      //println!("_");

      //println!("Best [Wi, bi] : {:?}", _wbi );
      //println!("c : {:?}", _c );   
      
        {
             //println!("---------------------TESTING-----------------------"); 
             //let mut test = vec![0.0f64; 2];
             //test[0] = 0.5;
             //test[1] = 0.9;
                       
             //println!("_");

             //println!("Real [Wi] = {:?}", nnet.weights);

             //println!("Real [bi] = {:?}", nnet.biases);

             //println!("_");
      
             //println!("[nnet] -> testing result Cos({:?}) --> {:?}", test, nnet.feed_forward(&test));              
        } 

        {
            test_water_quality();
        }

}

fn test_water_quality(){

        let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset.csv");
        let mut incols = Vec::new();
         //incols.push(0);
        incols.push(2);
        incols.push(3);
        incols.push(4);
        incols.push(5);
        incols.push(6);
        incols.push(7);
        incols.push(8);
       
        let mut outcols = Vec::new();
        outcols.push(1);
        
        let learn_part : usize = 127;
        
       let ds0 =  Dataset::read_from_csvfile(&path, &incols, &outcols);

       let (ds_learn, _ds_test) = ds0.get_shuffled().split_on_2(learn_part);

       //println!("dataset = {:?}", ds);

       println!("------------------WAETR QUALITY--------------");

       //println!("shuffled ataset = {:?}", ds.get_shuffled());            
        
       let layers:Vec<usize> = vec!{incols.len(), 4 , outcols.len()};
       let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
       let mut nnet = Neuralnet::new(layers, activations); 
                     
       //println!("In : {:?}", data_in);
       //println!("Out : {:?}", data_out);
       
       let p_size : usize = 70;
       let k_max : usize = 3000;
       let ub : f64 = 5.0;
       let lb : f64 = -5.0;
 
       let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
       
       let (_a, _wbi, _c) = eoann.learn();
       
       println!("_");
       
       println!("WQ - final learning error : RMSE = {:?}", _a );
       
       println!("_"); 

        {     

             //println!("Writing optimization curve ...");
             //let pathcrv =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_convergenceTrnd.csv");
             //let head = String::from("RMSE-Cnvergence_Trend");
             //let _error = Dataset::write_to_csv(&pathcrv, &Some(head), &_c);
             //println!("Writing optimization curve finish.");
        
             //println!("---------------------TESTING-----------------------"); 
         
           //println!("Real [Wi] = {:?}", nnet.weights);

            //println!("Real [bi] = {:?}", nnet.biases);

            //println!("_");
  
             //println!("[nnet] -> testing result [Ca], [Mg(]= {:?}) --> {:?}", test, nnet.feed_forward(&test));
         
             //let comuted_test = eoann.compute_out_for2(&ds_test.inputs);
            
             //println!("Writing test results ...");
             //let pathtest =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_test_results.csv");
             //let mut headers= Vec::new();
             //headers.push(String::from("Computed CE"));

             //let _error = Dataset::write_to_csv2(&pathtest, &Some(headers), &comuted_test);
             //println!("Writing test results finish.");
        
        
        //for rs in result.iter() {
         //   println!("{}", rs[0]);
        //}
       

    }

}

fn getdata_in(n : usize)->Vec<Vec<f64>> {
    let d=2;
    let mut positions = vec![vec![0.0f64; d]; n];
    let intervall01 = Uniform::from(0.0f64..0.9f64);
    let mut rng = rand::thread_rng();              
    //let mut _value = 0.0f64;
    for i in 0..n {
         for  j in 0..d {   
            // _value =f64::round(intervall01.sample(&mut rng)*100.0);                         
           
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
            positions[i][0] += data[i][j];            
        }
        positions[i][0]= f64::cos(positions[i][0]);
        //println!("cos = {}",positions[i][0]);
    }
    positions
}




