include!("solution.rs");
include!("sequential_eo.rs");
include!("benchmarks.rs");
include!("parallel_eo.rs");

//pub mod solution;
//pub mod sequential_eo;
//pub mod parallel_eo;
//pub mod benchmarks;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    fn seq_eo_test_f1() {
        let particuls = 12;
        let kmax = 100;
        let lb =-10.00;
        let ub = 10.00;
        let dim = 5;
         
        let (fbest, bestsol, convergcrv) = sequential_eo(particuls, kmax, lb,ub, dim, &fobj);
        assert_eq!(fbest.round(), 0.0f64);
        assert_eq!(bestsol.len(), dim);
        assert_eq!(convergcrv.len(), kmax);

    }

}
