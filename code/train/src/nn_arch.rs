use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Dropout, Linear, VarBuilder};
use candle_optimisers::Model;

const NDIHEDRALS: usize = 178;

#[derive(Debug)]
pub struct MyModel {
    dropout_1: Dropout,
    dropout_2: Dropout,
    conv1: Conv2d,
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
    train_input: Tensor,
    train_output: Tensor,
    test_input: Tensor,
    test_output: Tensor,
}

pub struct MySetupVars {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
}

impl MyModel {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vs: VarBuilder, setup: MySetupVars) -> Result<Self> {
        let dropout_1 = Dropout::new(0.2);
        let dropout_2 = Dropout::new(0.2);
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let conv1 = candle_nn::conv2d(1, 4, 3, conv_config, vs.pp("conv1"))?;
        let ln1 = candle_nn::linear(1424, 712, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(712, 356, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(356, NDIHEDRALS, vs.pp("ln3"))?;
        // let ln4 = candle_nn::linear(16, LABELS, vs.pp("ln4"))?;
        Ok(Self {
            dropout_1,
            dropout_2,
            conv1,
            ln1,
            ln2,
            ln3,
            // ln4,
            train_input: setup.train_data,
            train_output: setup.train_labels,
            test_input: setup.test_data,
            test_output: setup.test_labels,
        })
    }

    pub fn test_eval(&self) -> Result<f32> {
        let preds = self.forward(&self.test_input, false)?;
        angle_mse(&preds, &self.test_output)?.to_scalar()
    }
}

impl Model for MyModel {
    fn loss(&self) -> Result<Tensor> {
        let preds = self.forward(&self.train_input, true)?;
        angle_mse(&preds, &self.train_output)
    }
}

impl MyModel {
    fn forward(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        // let input = self.dropout.forward(&input, train)?;
        input
            .apply_t(&self.dropout_1, train)?
            .apply(&self.conv1)?
            .flatten_from(1)?
            .apply(&self.ln1)?
            .tanh()?
            .apply(&self.ln2)?
            .tanh()?
            .apply_t(&self.dropout_2, train)?
            .apply(&self.ln3)
    }
}

fn angle_mse(preds: &Tensor, actual: &Tensor) -> Result<Tensor> {
    let preds_cos = &preds.cos()?;
    let pred_sin = &preds.sin()?;

    let actual_cos = &actual.cos()?;
    let actual_sin = &actual.sin()?;

    // let mse_cos = loss::mse(preds_cos, actual_cos)?;
    // let mse_sin = loss::mse(pred_sin, actual_sin)?;
    ((preds_cos - actual_cos)?.sqr()? + (pred_sin - actual_sin)?.sqr()?)?.mean_all()
}
