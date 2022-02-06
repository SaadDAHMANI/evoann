const POP_SIZE : usize = 50;
const K_MAX : usize = 5;
const LB : f64 = -5.0;
const UB : f64 = 5.0;
const HL1 : usize = 5;

pub fn streamflow_forecast_loop(){
    
    let mut population = Vec::new();
   population.push(50usize);
   population.push(80usize);
   population.push(100usize);
   population.push(120usize);


   for p_size in population.into_iter() {

    println!("///////////////////////////////////////////////////----------POPULATIOON SIZE = {}", p_size);

     //let root = String::from("/home/sd/Documents/AppDev/Rust/evoann/data");
    let root = String::from("/home/roua/Documents/Rust/evoann/data");
    
    let path = format!("{}/{}", root, "Coxs3.csv"); 
     
    let mut incols = Vec::new();
    incols.push(2);
   // incols.push(3);
   // incols.push(4);
    
    let mut outcols = Vec::new();
    outcols.push(1);
    
   let learn_part : usize = 4868; 
    
   let ds0 =  Dataset::read_from_csvfile(&path, &incols, &outcols);

   let (ds_learn, ds_test) = ds0.get_shuffled().split_on_2(learn_part);
   let lcount = ds_learn.inputs.len();

   let ds_learn2 = ds_learn.clone();

   //println!("dataset = {:?}", ds);

   println!("----------------STREAMFLOW prediction-------------");

    //println!("In : {:?}", data_in);
    //println!("Out : {:?}", data_out);      
    //let p_size : usize = POP_SIZE;
   let k_max : usize = K_MAX;
   let ub : f64 = UB;
   let lb : f64 = LB;

    let chronos = Instant::now();

    let layers:Vec<usize> = vec!{incols.len(), HL1, outcols.len()};
   let annstruct = layers.clone();
   let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
   let mut nnet = Neuralnet::new(layers, activations); 
               
   
  //let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
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
        
         println!("Writing optimization curve ...");
         let pathcrv = format!("{}/Psize{}_{}", root, p_size ,"Coxs_convergenceTrnd_EO.csv");
                  
         let head = String::from("Coxs_RMSE-Cnvergence_Trend_EO");
         let _error = Dataset::write_to_csv(&pathcrv, &Some(head), &_c);
         println!("Writing optimization curve finish.");
    
         println!("---------------------TESTING-----------------------"); 
     
          //println!("[nnet] -> testing result [Ca], [Mg(]= {:?}) --> {:?}", test, nnet.feed_forward(&test));
         let computed_learn = eoann.compute_out_for2(&ds_learn2.inputs);  
         let computedlearn = convert2vector(&computed_learn);
         let observedlearn = convert2vector(&ds_learn2.outputs);   
         let _r2l =   Dataset::compute_determination_r2(&computedlearn, &observedlearn);

         println!("WQ - final Learning determination coef. : R2l = {:?}", _r2l);   
         println!("_");   
         
         println!("writing learning results .....");
              
         let pathlearn = format!("{}/Psize{}_{}", root, p_size, "Coxs_dataset_learn_results_EO.csv");             
                     
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
         
         let pathtest =format!("{}/Psize{}_{}",root ,p_size , "Coxs_dataset_test_results_EO.csv");
             
         
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
        let _path = format!("{}/{}", root, "Coxs3_mis_inputs.csv");
        
        let mut _incols = Vec::new();
        _incols.push(2);
        //_incols.push(3);
        //_incols.push(4);

        let mut _outcols = Vec::new();
        _outcols.push(1);
        
        let _dss =  Dataset::read_from_csvfile(&_path, &_incols, &_outcols);
        let _computed = eoann.compute_out_for2(&_dss.inputs);

        let mut _headers= Vec::new();
        _headers.push(String::from("Computed QC"));

        let _path_result = format!("{}/Psize{}_{}", root, p_size, "Coxs3_Outputs.csv");
        
        let _error = Dataset::write_to_csv2(&_path_result, &Some(_headers), &_computed);
         println!("Writing results is finish ...... OK.");        
    }
  }

 }
}

pub fn streamflow_forecast(){
    
    let root = String::from("/home/sd/Documents/AppDev/Rust/evoann/data");
    //let root = String::from("/home/roua/Documents/Rust/evoann/data");
    
    let path = format!("{}/{}", root, "Coxs3.csv"); 
 
    
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
    
   let layers:Vec<usize> = vec!{incols.len(), 8, outcols.len()};
   let annstruct = layers.clone();
   let activations:Vec<Activations> = vec!{Activations::Sigmoid, Activations::Linear};
   let mut nnet = Neuralnet::new(layers, activations); 
                 
   //println!("In : {:?}", data_in);
   //println!("Out : {:?}", data_out);      
   let p_size : usize = 50;
   let k_max : usize = 3500;
   let ub : f64 = 5.0;
   let lb : f64 = -5.0;

   //let mut eoann = SequentialEOTrainer::new(&mut nnet, ds_learn.inputs, ds_learn.outputs, p_size, k_max, lb, ub);
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
         let pathcrv = format!("{}/{}", root, "Coxs_convergenceTrnd_EO.csv");
                  
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
              
         let pathlearn = format!("{}/{}", root, "Coxs_dataset_learn_results_EO.csv");             
                     
         
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
         
         let pathtest =format!("{}/{}", root, "Coxs_dataset_test_results_EO.csv");
              
         
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
        let _path = format!("{}/{}", root , "Coxs3_inputs.csv");
        
        let mut _incols = Vec::new();
        _incols.push(2);
        _incols.push(3);
        _incols.push(4);

        let mut _outcols = Vec::new();
        _outcols.push(1);
        
        let _dss =  Dataset::read_from_csvfile(&_path, &_incols, &_outcols);
        let _computed = eoann.compute_out_for2(&_dss.inputs);

        let mut _headers= Vec::new();
        _headers.push(String::from("Computed QC"));

        let _path_result = format!("{}/{}", root, "Coxs3_Outputs.csv");
        
        let _error = Dataset::write_to_csv2(&_path_result, &Some(_headers), &_computed);
         println!("Writing results is finish ...... OK.");
        
    }

}

}
