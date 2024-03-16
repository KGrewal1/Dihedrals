use candle_core::{Result, Tensor};
use candle_nn::{ops::sigmoid, Conv2d, Linear, VarBuilder};
use candle_optimisers::Model;

#[derive(Debug)]
pub struct CheckCxModel {
    conv1: Conv2d,
    ln1: Linear,
    ln3: Linear,
}

impl CheckCxModel {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = candle_nn::conv2d(1, 1, 3, conv_config, vs.pp("conv1"))?;
        let ln1 = candle_nn::linear(356, 712, vs.pp("ln1"))?;
        let ln3 = candle_nn::linear(712, 1, vs.pp("ln3"))?;
        // let ln4 = candle_nn::linear(16, LABELS, vs.pp("ln4"))?;
        Ok(Self { conv1, ln1, ln3 })
    }
}

impl Model for CheckCxModel {
    fn loss(&self) -> Result<Tensor> {
        panic!();
    }
}

impl CheckCxModel {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // let input = self.dropout.forward(&input, train)?;
        input
            .apply(&self.conv1)?
            .flatten_from(1)?
            .apply(&self.ln1)?
            .tanh()?
            .apply(&self.ln3)?
            .apply(&sigmoid)
    }
}
