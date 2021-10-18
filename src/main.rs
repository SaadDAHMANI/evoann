
include!("neuralnet.rs");
include!("activations.rs");
include!("eo_trainer.rs");

fn main() {
    println!("Hello, world!");

    let layers:Vec<usize> = vec!{2,5,2};
    let activations:Vec<Activations> = vec!{Activations::TanH, Activations::Linear };
    let mut nn = Neuralnet::new(layers, activations);
    
    //let trainer = Trainer {
     //   neuralnet : nn,
    //};   
    
    let mut w = vec![0.0f32; nn.get_weights_count()];

    for i in 0..w.len() {
        w[i]=i as f32;
    }
    

    println!("W before update = {:?}", nn.weights);

    nn.update_weights(&w);

    println!("W after update = {:?}", nn.weights);
    
    let mut b = vec![0.0f32; nn.get_biases_count()];

    for i in 0..b.len() {
        b[i]=i as f32;
    }
    
    println!("biases before update = {:?}", nn.biases);

    nn.update_biases(&b);

    println!("biases after update = {:?}", nn.biases);








    
}
