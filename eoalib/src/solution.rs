
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Solution {
    index : usize,
    c : Vec<f64>,
    fitness : f64,
}

impl Solution {
    pub fn new(indx : usize, dim : usize) -> Solution {
        let sln = Solution {
            index : indx,
            c : vec![0.0f64; dim],
            fitness : -1.0,
        };
        sln
    }

    pub fn to_string(&self)->String {
        format!("-> {}, {:?}, >> F: {}", self.index, self.c, self.fitness)
    }

}