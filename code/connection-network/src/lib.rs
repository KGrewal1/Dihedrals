use candle_core::{Result, Tensor};
use candle_nn::{ops::sigmoid, Linear, Module, VarBuilder};

const NDIHEDRALS: usize = 178;
const NINPUTS: usize = 2 * NDIHEDRALS;
const NHIDDEN: usize = 356;

#[derive(Debug)]
pub struct CheckCxModel {
    ln1: Linear,
    ln2: Linear,
}

impl CheckCxModel {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(NINPUTS, NHIDDEN, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(NHIDDEN, 1, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }
}

impl Module for CheckCxModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input
            .flatten_from(1)?
            .apply(&self.ln1)?
            .tanh()?
            .apply(&self.ln2)?
            .apply(&sigmoid)
    }
}
