use std::io;

use csv;
use serde;
use serde::de::DeserializeOwned;
use serde::Deserialize;

#[derive(Hash, Eq, PartialEq, Debug, Ord, PartialOrd, Deserialize)]
struct Type1 {
    id: u8,
    name: String,
}

#[derive(Hash, Eq, PartialEq, Debug, Ord, PartialOrd, Deserialize)]
struct Type2 {
    id: u8,
    name: String,
}

fn parse_csv<D: DeserializeOwned, R: io::Read>(rdr: R) -> csv::Result<Vec<D>> {
    csv::Reader::from_reader(rdr).into_deserialize().collect()
}

pub fn main_csvrw() -> Result<(), Box<dyn std::error::Error>> {
    let data1 = "\
id,name
1,\"X\"
2,\"Y\"
";
    let data2 = "\
id,name
1,\"A\"
2,\"B\"
";

    let records1: Vec<Type1> = parse_csv(data1.as_bytes())?;
    let records2: Vec<Type2> = parse_csv(data2.as_bytes())?;
    
    println!("{:?}", records1);
    println!("{:?}", records2);
    
    Ok(())
}


pub fn csvrw(path : &str) -> Result<(), Box<dyn std::error::Error>> {
    let data1 = "\
id,name
1,\"X\"
2,\"Y\"
";
    let data2 = "\
id,name
1,\"A\"
2,\"B\"
";

    let records1: Vec<Type1> = parse_csv(data1.as_bytes())?;
    let records2: Vec<Type2> = parse_csv(data2.as_bytes())?;
    
    println!("{:?}", records1);
    println!("{:?}", records2);
    
    let mut reader = csv::Reader::from_path(path)?;
    
    let headers = reader.headers()?;
    
    println!("Headers :  {:?}", headers);

    for result in reader.records() {

        let record = result?;

        if record.is_empty()==false {
            let l = record.len();
            for i in 0..l {
                  print!("{:?}", &record[i]);
                  print!(" ");
                
             }
             println!("_");
        }
    }







    Ok(())
}