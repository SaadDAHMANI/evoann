
include!("neuralnet.rs");
include!("activations.rs");
//include!("eo_trainer.rs");
include!("dataset.rs");
include!("solution.rs");
include!("common.rs");
include!("seq_eo_trainer.rs");
include!("seq_pso_trainer.rs");

//extern crate eoalib;
//se eoalib::*;
use std::error::Error;
use std::time::{Duration, Instant};



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
            streamflow_forecast();
           //test_water_quality();
           // test_water_quality_loop();           
            
        }

}


fn streamflow_forecast(){

    //let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/2Qm1.csv");
    let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs3.csv");
    
    let mut incols = Vec::new();
    incols.push(2);
    incols.push(3);
    incols.push(4);
    
    let mut outcols = Vec::new();
    outcols.push(1);
    
    let learn_part : usize = 4868; 
    
   let ds0 =  Dataset::read_from_csvfile(&path, &incols, &outcols);

   let (ds_learn, ds_test) = ds0.get_shuffled().split_on_2(learn_part);
   let lcount = ds_learn.inputs.len();

   let ds_learn2 = ds_learn.clone();

   //println!("dataset = {:?}", ds);

   println!("----------------STREAMFLOW prediction-------------");

   let chronos = Instant::now();

   //println!("shuffled ataset = {:?}", ds.get_shuffled());            
    
   let layers:Vec<usize> = vec!{incols.len(), 5, outcols.len()};
   let annstruct = layers.clone();
   let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
   let mut nnet = Neuralnet::new(layers, activations); 
                 
   //println!("In : {:?}", data_in);
   //println!("Out : {:?}", data_out);      
   let p_size : usize = 30;
   let k_max : usize = 1000;
   let ub : f64 = 5.0;
   let lb : f64 = -5.0;

   // let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
   let mut eoann = SequentialPSOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
   // set EO params ----------
   // eoann.a1 = 2.0; 
   // eoann.a2 = 1.0;  
   // eoann.gp = 0.5;
   //-------------------------
           // set EO params ----------
   eoann.c1 = 2.0; 
   eoann.c2 = 2.0;        
   //-------------------------
  

   let (_a, _wbi, _c) = eoann.learn();
   
   println!("_Learning items count : {:?}",lcount);
   println!("*Population-size : {},  *ANN-Struct :{:?}", p_size, annstruct);

   println!("_");
   
   println!("WQ - final Learning error : RMSEl = {:?}", _a );
   
   println!("_"); 
    {     
        
         //println!("Writing optimization curve ...");
         let pathcrv =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs_convergenceTrnd_EO.csv");
         //let pathcrv =String::from("/home/roua/Documents/Rust/evoann/data/Coxs_convergenceTrnd_EO.csv");
         
         let head = String::from("Coxs_RMSE-Cnvergence_Trend_EO");
         let _error = Dataset::write_to_csv(&pathcrv, &Some(head), &_c);
         println!("Writing optimization curve finish.");
    
         //println!("---------------------TESTING-----------------------"); 
     
       //println!("Real [Wi] = {:?}", nnet.weights);

        //println!("Real [bi] = {:?}", nnet.biases);

        //println!("_");

         //println!("[nnet] -> testing result [Ca], [Mg(]= {:?}) --> {:?}", test, nnet.feed_forward(&test));
         let computed_learn = eoann.compute_out_for2(&ds_learn2.inputs);  
         let computedlearn = convert2vector(&computed_learn);
         let observedlearn = convert2vector(&ds_learn2.outputs);   
         let _r2l =   Dataset::compute_determination_r2(&computedlearn, &observedlearn);

         println!("WQ - final Learning determination coef. : R2l = {:?}", _r2l);   
         println!("_");   
         
         println!("writing learning results .....");
              
         let pathlearn  = String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs_dataset_learn_results_EO.csv");             
         //let pathlearn  = String::from("/home/roua/Documents/Rust/evoann/data/Coxs_dataset_learn_results_EO.csv");             
         
         let mut headers = Vec::new();
         headers.push(String::from("Computed CE"));
         let _error = Dataset::write_to_csv2(&pathlearn, &Some(headers), &computed_learn);
         println!("Writing learning results ......OK");

        
         let computed_test = eoann.compute_out_for2(&ds_test.inputs);

         let computed = convert2vector(&computed_test);
         let observed = convert2vector(&ds_test.outputs);
         let rmse_test = Dataset::compute_rmse(&computed, &observed);
         println!("WQ - final Testing error : RMSEt = {:?}", rmse_test);

         let _r2t =   Dataset::compute_determination_r2(&computed, &observed);
         println!("WQ - final Testing determination coef : R2t = {:?}", _r2t);   
        
         println!("Writing test results ......");
         
         let pathtest =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs_dataset_test_results_EO.csv");
         //let pathtest =String::from("/home/roua/Documents/Rust/evoann/data/Coxs_dataset_test_results_EO.csv");
         
         
         let mut headers= Vec::new();
         headers.push(String::from("Computed CE-Test"));
         println!("test elements :  {:?}", computed_test.len());   
         let _error = Dataset::write_to_csv2(&pathtest, &Some(headers), &computed_test);
         println!("Writing test results finish ...... OK.");
    
    
    //for rs in result.iter() {
     //   println!("{}", rs[0]);
    //}
   
    let duration = chronos.elapsed();
    println!("End computation in : {:?}.", duration);

    {
        let _path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs3_inputs.csv");
        //let _path =String::from("/home/roua/Documents/Rust/evoann/data/Coxs3_inputs.csv");

        let mut _incols = Vec::new();
        _incols.push(2);
        _incols.push(3);
        
        let mut _outcols = Vec::new();
        _outcols.push(1);
        
        let _dss =  Dataset::read_from_csvfile(&_path, &_incols, &_outcols);
        let _computed = eoann.compute_out_for2(&_dss.inputs);

        let mut _headers= Vec::new();
        _headers.push(String::from("Computed QC"));

        let _path_result =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/Coxs_dataset_test_results_EO.csv");
        //let _path_result =String::from("/home/roua/Documents/Rust/evoann/data/Coxs_dataset_test_results_EO.csv");

        let _error = Dataset::write_to_csv2(&_path_result, &Some(_headers), &_computed);
         println!("Writing results is finish ...... OK.");
        
    }

}

}


fn test_water_quality(){

        //let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/2Qm1.csv");
        let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset.csv");
        
        let mut incols = Vec::new();
        //incols.push(1);
        incols.push(2);
        incols.push(3);
        incols.push(4);
        incols.push(5);
        incols.push(6);
        incols.push(7);
        incols.push(8);
       
        let mut outcols = Vec::new();
        outcols.push(1);
        
        let learn_part : usize = 127; //297;
        
       let ds0 =  Dataset::read_from_csvfile(&path, &incols, &outcols);

       let (ds_learn, ds_test) = ds0.get_shuffled().split_on_2(learn_part);
       let lcount = ds_learn.inputs.len();

       let ds_learn2 = ds_learn.clone();

       //println!("dataset = {:?}", ds);

       println!("------------------WAETR QUALITY--------------");

       let chronos = Instant::now();

       //println!("shuffled ataset = {:?}", ds.get_shuffled());            
        
       let layers:Vec<usize> = vec!{incols.len(), 5, outcols.len()};
       let annstruct = layers.clone();
       let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
       let mut nnet = Neuralnet::new(layers, activations); 
                     
       //println!("In : {:?}", data_in);
       //println!("Out : {:?}", data_out);      
       let p_size : usize = 30;
       let k_max : usize = 3000;
       let ub : f64 = 5.0;
       let lb : f64 = -5.0;

       let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
       //let mut eoann = SequentialPSOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
       // set EO params ----------
        eoann.a1 = 2.0; 
        eoann.a2 = 1.0;  
        eoann.gp = 0.5;
       //-------------------------
               // set EO params ----------
       //eoann.c1 = 2.0; 
       //eoann.c2 = 2.0;        
       //-------------------------
      

       let (_a, _wbi, _c) = eoann.learn();
       
       println!("_Learning items count : {:?}",lcount);
       println!("*Population-size : {},  *ANN-Struct :{:?}", p_size, annstruct);

       println!("_");
       
       println!("WQ - final Learning error : RMSEl = {:?}", _a );
       
       println!("_"); 
        {     
            
             //println!("Writing optimization curve ...");
             let pathcrv =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_convergenceTrnd_EO.csv");
             let head = String::from("RMSE-Cnvergence_Trend_EO");
             let _error = Dataset::write_to_csv(&pathcrv, &Some(head), &_c);
             println!("Writing optimization curve finish.");
        
             //println!("---------------------TESTING-----------------------"); 
         
           //println!("Real [Wi] = {:?}", nnet.weights);

            //println!("Real [bi] = {:?}", nnet.biases);

            //println!("_");
  
             //println!("[nnet] -> testing result [Ca], [Mg(]= {:?}) --> {:?}", test, nnet.feed_forward(&test));
             let computed_learn = eoann.compute_out_for2(&ds_learn2.inputs);  
             let computedlearn = convert2vector(&computed_learn);
             let observedlearn = convert2vector(&ds_learn2.outputs);   
             let _r2l =   Dataset::compute_determination_r2(&computedlearn, &observedlearn);

             println!("WQ - final Learning determination coef. : R2l = {:?}", _r2l);   
             println!("_");   
             
             println!("writing learning results .....");
                  
             let pathlearn  = String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_learn_results_EO.csv");             
             let mut headers = Vec::new();
             headers.push(String::from("Computed CE"));
             let _error = Dataset::write_to_csv2(&pathlearn, &Some(headers), &computed_learn);
             println!("Writing learning results ......OK");
    
            
             let computed_test = eoann.compute_out_for2(&ds_test.inputs);

             let computed = convert2vector(&computed_test);
             let observed = convert2vector(&ds_test.outputs);
             let rmse_test = Dataset::compute_rmse(&computed, &observed);
             println!("WQ - final Testing error : RMSEt = {:?}", rmse_test);

             let _r2t =   Dataset::compute_determination_r2(&computed, &observed);
             println!("WQ - final Testing determination coef : R2t = {:?}", _r2t);   
            
             println!("Writing test results ......");
             let pathtest =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_test_results_EO.csv");
             let mut headers= Vec::new();
             headers.push(String::from("Computed CE-Test"));
             println!("test elements :  {:?}", computed_test.len());   
             let _error = Dataset::write_to_csv2(&pathtest, &Some(headers), &computed_test);
             println!("Writing test results finish ...... OK.");
        
        
        //for rs in result.iter() {
         //   println!("{}", rs[0]);
        //}
       
        let duration = chronos.elapsed();
        println!("End computation in : {:?}.", duration);

    }

}


fn test_water_quality_loop(){

    let mut pop_sizes = Vec::new();
    pop_sizes.push(30usize);
    pop_sizes.push(50usize);          
    pop_sizes.push(70usize);
    pop_sizes.push(100usize);

    let mut result_values = Vec::new();
    let mut hidden_layer : usize=0; 

    let rootpath = String::from("/home/sd/Documents/AppDev/Rust/evoann/data/"); 
    
    for psize in pop_sizes.into_iter() {
     
    //let path =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/2Qm1.csv");
    let path =format!("{}dataset.csv",rootpath);
    
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

    let (ds_learn, ds_test) = ds0.get_shuffled().split_on_2(learn_part);
    let lcount = ds_learn.inputs.len();

    let ds_learn2 = ds_learn.clone();

   //println!("dataset = {:?}", ds);

   println!("------------------WAETR QUALITY--------------");

   let chronos = Instant::now();

   //println!("shuffled ataset = {:?}", ds.get_shuffled());  
  
     let layers:Vec<usize> = vec!{incols.len(), 11, outcols.len()};
     let annstruct = layers.clone();
     hidden_layer=layers[1].clone();

     let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
     let mut nnet = Neuralnet::new(layers, activations); 

               
   //println!("In : {:?}", data_in);
   //println!("Out : {:?}", data_out);
   
   let p_size : usize = psize;
   let k_max : usize = 3000;
   let ub : f64 = 5.0;
   let lb : f64 = -5.0;

   //let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
   let mut eoann = SequentialPSOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
   // set EO params ----------
   //eoann.a1 = 2.0; 
   //eoann.a2 = 1.0;  
   //eoann.gp = 0.5;
   //-------------------------
          // set EO params ----------
  eoann.c1 = 2.0; 
  eoann.c2 = 2.0;  
  //eoann.gp = 0.5;
  //-------------------------
  
   
   let (_a, _wbi, _c) = eoann.learn();
   
   println!("_Learning items count : {:?}",lcount);
   println!("*Population-size : {},  *ANN-Struct :{:?}", p_size, annstruct);

   println!("_");
   
   println!("WQ - final Learning error : RMSEl = {:?}", _a );
   
   println!("_"); 
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
         let computed_learn = eoann.compute_out_for2(&ds_learn2.inputs);  
         let computedlearn = convert2vector(&computed_learn);
         let observedlearn = convert2vector(&ds_learn2.outputs);   
         let _r2l =   Dataset::compute_determination_r2(&computedlearn, &observedlearn);

         println!("WQ - final Learning determination coef. : R2l = {:?}", _r2l);   
         println!("_");   
     
         let computed_test = eoann.compute_out_for2(&ds_test.inputs);

         let computed = convert2vector(&computed_test);
         let observed = convert2vector(&ds_test.outputs);
         let rmse_test = Dataset::compute_rmse(&computed, &observed);
         println!("WQ - final Testing error : RMSEt = {:?}", rmse_test);

         let _r2t =   Dataset::compute_determination_r2(&computed, &observed);
         println!("WQ - final Testing determination coef : R2t = {:?}", _r2t); 
         
         let mut result = vec![0.0f64; 4];
         result[0]= _a;
         result[1]= rmse_test.unwrap();
         result[2]= _r2l.unwrap();
         result[3]= _r2t.unwrap();
         result_values.push(result);
        
         //println!("Writing test results ...");
         //let pathtest =String::from("/home/sd/Documents/AppDev/Rust/evoann/data/dataset_test_results.csv");
         //let mut headers= Vec::new();
         //headers.push(String::from("Computed CE"));

         //let _error = Dataset::write_to_csv2(&pathtest, &Some(headers), &comuted_test);
         //println!("Writing test results finish.");
          
   

    //for rs in result.iter() {
     //   println!("{}", rs[0]);
    //}
   

    let duration = chronos.elapsed();
    println!("End computation in : {:?}.", duration);

    }
    
    let file_path =format!("{}result_HL{:?}_.csv", rootpath, hidden_layer);
         let mut headers = Vec::new();
         headers.push(String::from("RMSEl"));         
         headers.push(String::from("RMSEt"));
         headers.push(String::from("R2l"));
         headers.push(String::from("R2t"));
         let _error = Dataset::write_to_csv2(&file_path, &Some(headers), &result_values);  
    println!("End saving.");

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

fn convert2vector(data : &Vec<Vec<f64>>)-> Vec<f64> {
    let mut vect = Vec::new();
    for i in 0..data.len() {
        vect.push(data[i][0]);
    }
    vect
}


pub enum TrainerAlgo{
    EO,
    PSO,
}



