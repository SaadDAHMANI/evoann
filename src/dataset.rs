//use std::io;

use csv;
use csv::Writer;

//use serde::de::DeserializeOwned;
//use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Dataset {
    //inputs_headers : &'a mut Vec<String>,
    //outputs_headers : &'a mut Vec<String>,
    inputs : Vec<Vec<f64>>,
    outputs : Vec<Vec<f64>>, 
    file_path : Option<String>,    
}

impl Dataset{

    pub fn new(inputs : Vec<Vec<f64>>, outputs :Vec<Vec<f64>>, file : Option<String>)-> Dataset {
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
            file_path : Some(flepath),
        };
        return ds;
    }

    pub fn readall_from_file(path : &str)-> Result< Vec<Vec<f64>>, Box<dyn Error>> {
    
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
   
    pub fn write_to_csv(path : &String, header : &Option<String>, data : &Vec<f64>)-> Result<(), Box<dyn Error> > {
        let mut wtr = Writer::from_path(path)?;

        match header {
            Some(header) => wtr.write_record(&[header])?,
            None => (),
        };

        for i in 0..data.len() {
             wtr.write_record(&[data[i].to_string()])?;
        }

        wtr.flush()?;
        
        Ok(())   
    }

    pub fn write_to_csv2(path : &String, headers : &Option<Vec<String>>, data : &Vec<Vec<f64>>)-> Result<(), Box<dyn Error> > {
        let mut wtr = Writer::from_path(path)?;

        match headers {
            Some(header) => wtr.write_record(header.iter())?,
            None => (),
        };

        for row in data {
             let cols_str: Vec<_> = row.iter().map(ToString::to_string).collect();   
             //let line = cols_str.join(",");
             wtr.write_record(cols_str.iter())?;
        }

        wtr.flush()?;
        
        Ok(())   
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

    pub fn split_on_2(&self, firstelemntscount : usize)->(Dataset, Dataset) {
        let totalcount = self.inputs.len();

        let firstcount = usize::min(firstelemntscount, totalcount);
        
        let mut datain1 = Vec::new();
        let mut datain2 = Vec::new();
        let mut dataout1 = Vec::new();
        let mut dataout2 = Vec::new();

        for i in 0..firstcount {
            datain1.push(self.inputs[i].clone());
            dataout1.push(self.outputs[i].clone());
        }

        for i in firstcount..totalcount {
            datain2.push(self.inputs[i].clone());
            dataout2.push(self.outputs[i].clone());
        } 

        
        let ds1 = Dataset::new(datain1, dataout1, None);
        let ds2 = Dataset::new(datain2, dataout2, None);

        (ds1, ds2)
    }  

    pub fn get_rmse(ds1 : &Vec<f64>, ds2 : &Vec<f64>)->f64 {
         let mut rmse : f64 =0.0;
         let count = usize::min(ds1.len(), ds2.len());
         
         for i in 0..count {
            rmse += f64::powi(ds1[i]- ds2[i], 2);
         }
         rmse = rmse /count as f64;
         rmse = f64::sqrt(rmse); 
         rmse
    }
}