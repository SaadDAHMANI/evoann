//use std::io;

use csv;
//use serde::de::DeserializeOwned;
//use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Dataset {
    //inputs_headers : &'a mut Vec<String>,
    //outputs_headers : &'a mut Vec<String>,
    inputs : Vec<Vec<f64>>,
    outputs : Vec<Vec<f64>>, 
    file_path : String,
    
}

impl Dataset{

    pub fn new(inputs : Vec<Vec<f64>>, outputs :Vec<Vec<f64>>, file : String)-> Dataset {
        Dataset{
            inputs : inputs,
            outputs : outputs, 
            file_path : file,            
        }
    }
    
    pub fn read_from_csvfile(path : &String, in_cols : &Vec<usize>, out_cols : &Vec<usize>)->Dataset{
            
        let origing_data = Dataset::readall_from_file(path);

        let data = match origing_data {
            Ok(data) => data,
            Err(error) => panic!("Problem was found : {:?}", error),
        };

        let mut input = Vec::new();
        let mut output = Vec::new();

        for i in 0..data.len() {
            let mut row_in = Vec::new();
            let mut row_out = Vec::new();

            for j in in_cols.iter(){
                if *j< data[i].len() {
                    row_in.push(data[i][*j]);
                }  
            }

            for k in out_cols.iter() {
                if *k < data[i].len() {
                    row_out.push(data[i][*k]);
                }
            }
 
            input.push(row_in);
            output.push(row_out);
        }
        
        let flepath = path.clone();
        let ds = Dataset {
            inputs : input,
            outputs : output,
            file_path : flepath,
        };
        return ds;
    }

    fn readall_from_file(path : &str)-> Result< Vec<Vec<f64>>, Box<dyn Error>> {
    
         let mut reader = csv::Reader::from_path(path)?;
    
         let headers = reader.headers()?;
    
         println!("Headers :  {:?}", headers);

         let mut data = Vec::new();

        for result in reader.records() {

             let record = result?;

             if record.is_empty()==false {
                 let l = record.len();
                 let mut row = vec![0.0f64; l];

                 for i in 0..l {
                 row[i] = record[i].parse()?; 
                }
            data.push(row)            
            }       
        }
         Ok(data)
    }
   
    ///
    /// shuffled data in the intervalle [0, 1] 
    /// 
    pub fn get_shuffled(&self)->Dataset {
        let mut maxin = Vec::new();

        if self.inputs.len()> 0 {
             maxin = self.inputs[0].clone();        
        }        

        let mut maxout = Vec::new();
        if self.outputs.len()> 0 {
             maxout = self.outputs[0].clone();
        }
        
        let icountin = self.inputs.len();
        let jcountin = self.inputs[0].len(); 

        for j in 0.. jcountin {
            for i in 0..icountin {
               if maxin[j] < self.inputs[i][j] {
                   maxin[j]=self.inputs[i][j];
               }  
            }   
        }

        let icountout = self.outputs.len();
        let jcountout = self.outputs[0].len(); 

        for j in 0.. jcountout {
            for i in 0..icountout {
               if maxout[j] < self.outputs[i][j] {
                   maxout[j]=self.outputs[i][j];
               }  
            }   
        }

        let mut shuffled = self.clone();
        for i in 0..icountin {
            for j in 0.. jcountin {
                if maxin[j] != 0.0 {
                    shuffled.inputs[i][j]= shuffled.inputs[i][j]/maxin[j]; 
                }                               
            }
        }
        
        for i in 0..icountout {
            for j in 0.. jcountout {
                if maxout[j] != 0.0 {
                    shuffled.outputs[i][j]= shuffled.outputs[i][j]/maxout[j]; 
                }                               
            }
        }

        return shuffled;
    } 

     ///
    /// shuffled data in the intervalle [0, 0.9] 
    /// 
    pub fn get_shuffled_09(&self)->Dataset {
        let mut maxin = Vec::new();
        let mut minin = Vec::new();

        if self.inputs.len()> 0 {
             maxin = self.inputs[0].clone();
             minin =self.inputs[0].clone();        
        }        

        let mut maxout = Vec::new();
        let mut minout = Vec::new();
        if self.outputs.len()> 0 {
             maxout = self.outputs[0].clone();
             minout = self.outputs[0].clone();
        }
        
        let icountin = self.inputs.len();
        let jcountin = self.inputs[0].len(); 

        // search min and max values
        for j in 0.. jcountin {
            for i in 0..icountin {
               if maxin[j] < self.inputs[i][j] {
                   maxin[j] = self.inputs[i][j];
               }
               
               if minin[j] > self.inputs[i][j] {
                   minin[j] = self.inputs[i][j]
               }   
            }   
        }

        let icountout = self.outputs.len();
        let jcountout = self.outputs[0].len(); 

         // search min and max values
        for j in 0.. jcountout {
            for i in 0..icountout {
               if maxout[j] < self.outputs[i][j] {
                     maxout[j]=self.outputs[i][j];
               }  

               if minout[j] > self.outputs[i][j] {
                     minout[j]=self.outputs[i][j];
                }  
            }   
        }

        let mut shuffled = self.clone();
        for i in 0..icountin {
            for j in 0.. jcountin {
                if maxin[j] != 0.0 {
                    shuffled.inputs[i][j]= 0.9 *(shuffled.inputs[i][j]-minin[j])/(maxin[j]-minin[j]); 
                }                               
            }
        }
        
        for i in 0..icountout {
            for j in 0.. jcountout {
                if maxout[j] != 0.0 {
                    shuffled.outputs[i][j]= 0.9*(shuffled.outputs[i][j]-minout[j])/(maxout[j]-minout[j]); 
                }                               
            }
        }

        return shuffled;
    } 



}