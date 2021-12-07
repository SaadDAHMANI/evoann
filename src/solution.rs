#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Solution {
    index : usize,
    c : Vec<f64>,
    fitness : f64,
}

impl Solution {
    
    //
    // Initialize a new solution with fitness value = f64::MAX
    //
    pub fn new(indx : usize, dim : usize) -> Solution {
        let sln = Solution {
            index : indx,
            c : vec![0.0f64; dim],
            fitness : f64::MAX,
        };
        sln
    }

    pub fn newf(indx : usize, dim : usize, fit : f64) -> Solution {
        let sln = Solution {
            index : indx,
            c : vec![0.0f64; dim],
            fitness : fit,
        };
        sln
    }
    
    pub fn to_string(&self)->String {
        format!("-> {}, {:?}, >> F: {:?}", self.index, self.c, self.fitness)
    }

}