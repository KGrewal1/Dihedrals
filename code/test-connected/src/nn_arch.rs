use candle_core::{Result, Tensor};
use candle_nn::{ops::sigmoid, Conv2d, Linear, VarBuilder};

#[derive(Debug)]
pub struct CheckCxModel {
    conv1: Conv2d,
    ln1: Linear,
    ln2: Linear,
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
        let ln2 = candle_nn::linear(712, 1, vs.pp("ln2"))?;
        Ok(Self { conv1, ln1, ln2 })
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
            .apply(&self.ln2)?
            .apply(&sigmoid)
    }
}
