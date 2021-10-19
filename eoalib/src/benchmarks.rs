//----------------------------------------------------------------------------------
// Implemented in Rust programming language by :
// Saad Dahmani (sd.dahmani2000@gmail.com; s.dahmani@univ-bouira.dz)
//----------------------------------------------------------------------------------

pub fn f1(x : &Vec<f64>)-> f64 {
   
    let sum = x.iter().map(|s| s.powi(2)).sum();
    //for value in x.iter(){
    //sum += value.powi(2);
    //}  
    sum
}

pub fn f2(x: &Vec<f64>)-> f64 {
    //o=sum(abs(x))+prod(abs(x));
    let mut sum : f64 = 0.0;
    let mut prod : f64 =1.0;

     for value in x.iter(){
         sum += value.abs();
         prod *=value.abs();
     }  
       sum + prod    
}



