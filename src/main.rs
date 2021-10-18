
include!("neuralnet.rs");
include!("activations.rs");
include!("eo_trainer.rs");

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{3,2,1};
    let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
    let mut nn = Neuralnet::new(layers, activations);
    
    //let trainer = Trainer {
     //   neuralnet : nn,
    //};   
    
    let w = vec![0.0f32; nn.get_weights_count()];

    println!("W before update = {:?}", nn.weights);

    nn.update_weights(&w);


    println!("W after update = {:?}", nn.weights);







    
}
