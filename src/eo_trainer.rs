pub struct Trainer {

    neuralnet : Neuralnet,

}
impl Trainer {

    fn get_weights_count(&self)-> usize {

     let mut count : usize =0;
     
     for i in 1..self.neuralnet.layers.len() {
        count += self.neuralnet.layers[i-1]* self.neuralnet.layers[i]; 
     }
     count
    }

    fn get_biases_count(&self)->usize {
     
        let mut count : usize =0;
     
        for i in 1..self.neuralnet.layers.len() {
             count += self.neuralnet.layers[i]; 
        }
        count
    }

    pub fn weights_biases_count(&self)-> usize {
       self.get_weights_count()+ self.get_biases_count()
    }



    

}


#[cfg(test)]
mod trainer_tests {
 use super::*;

    #[test]
    fn get_weights_count_test1(){

         let layers:Vec<usize> = vec!{3,4,2};
         let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
         let nn = Neuralnet::new(layers, activations);
         
         let trainer = Trainer {
             neuralnet : nn,
         };   
         
         assert_eq!(trainer.get_weights_count(), 12+8);
    }

    #[test]
    fn get_biases_count_test1() {

        let layers:Vec<usize> = vec!{5,10,3};
         let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
         let nn = Neuralnet::new(layers, activations);
         
         let trainer = Trainer {
             neuralnet : nn,
         };   
         
         assert_eq!(trainer.get_biases_count(), 10+3);
    }


    #[test]
    fn weights_biases_count_test1(){

         let layers:Vec<usize> = vec!{4,8,1};
         let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
         let nn = Neuralnet::new(layers, activations);
         
         let trainer = Trainer {
             neuralnet : nn,
         };   
         
         assert_eq!(trainer.weights_biases_count(), (8*4+8*1+8+1));
    }

    #[test]
    fn weights_biases_count_test2(){

         let layers:Vec<usize> = vec!{4,8,3,1};
         let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
         let nn = Neuralnet::new(layers, activations);
         
         let trainer = Trainer {
             neuralnet : nn,
         };   
         
         assert_eq!(trainer.weights_biases_count(), (8*4+8*3+ 3*1+8+3+1));
    }



}
