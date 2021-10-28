use std::io;

use csv;
use serde::de::DeserializeOwned;
use serde::Deserialize;

//#[derive(Debug, Clone)]
pub struct Dataset<'a> {
    //inputs_headers : &'a mut Vec<String>,
    //outputs_headers : &'a mut Vec<String>,
    inputs : &'a mut Vec<Vec<f64>>,
    outputs : &'a mut Vec<Vec<f64>>, 
    file_path : String,
    
}

impl<'a> Dataset<'a> {

    pub fn new(inputs : &'a mut Vec<Vec<f64>>, outputs : &'a mut Vec<Vec<f64>>, file : String)-> Dataset<'a> {
        Dataset{
            inputs : inputs,
            outputs : outputs, 
            file_path : file,            
        }
    }

    pub fn read_from_csvfile(path : &str, in_cols : &Vec<usize>, out_cols : &Vec<usize>) {
            
        let origing_data = read_from_file(path);

        let data = match origing_data {
            Ok(data) => data,
            Err(error) => panic!("Problem was found : {:?}", error),
        };

        let icount = data.len();
        let mut jcount : usize =0;

        if icount > 0 {
            jcount = data[0].len();
        }
        
        




    }


    fn read_from_file(path : &str)-> Result< Vec<Vec<f64>>, Box<dyn Error>> {
    
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





}