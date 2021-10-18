
include!("neuralnet.rs");
include!("activations.rs");
include!("eo_trainer.rs");

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{4,2,1};
    let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
    let mut nn = Neuralnet::new(layers, activations);
    
    //let trainer = Trainer {
     //   neuralnet : nn,
    //};   
    
    let mut wb = vec![0.0f64; nn.get_weights_biases_count()];

    for i in 0..wb.len() {
        wb[i] = i as f64;
    }
    

    println!("W before update = {:?}", nn.weights);

    println!("biases before update = {:?}", nn.biases);

    nn.update_weights_biases(&wb);

    println!("W after update = {:?}", nn.weights);
    
    println!("biases after update = {:?}", nn.biases);








    
}
