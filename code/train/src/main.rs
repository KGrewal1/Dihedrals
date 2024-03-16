use candle_core::DType;
use candle_nn::Optimizer;
use candle_optimisers::{
    nadam::{NAdam, ParamsNAdam},
    Model,
};
use env_logger::{Builder, Target};
use log::{info, warn, LevelFilter};
use setup_training::setup_training;

mod nn_arch;
mod setup_training;

const DATATYPE: DType = DType::F32;
fn main() -> anyhow::Result<()> {
    let mut builder = Builder::new();
    builder.format_target(false);
    builder.target(Target::Stdout);
    builder.filter(None, LevelFilter::Info);
    builder.init();

    let (model, varmap) = setup_training()?;
    model.loss()?;

    let adam_params = ParamsNAdam {
        lr: 0.0001,
        beta_1: 0.9,
        beta_2: 0.999,
        eps: 1e-8,
        weight_decay: None,
        momentum_decay: 0.004,
    };
    let mut optimiser = NAdam::new(varmap.all_vars(), adam_params)?;
    println!("Starting training");
    let mut initial_loss = model.loss()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    for i in 1..=1_000_000 {
        let loss = model.loss()?;
        optimiser.backward_step(&loss)?;
        if i % 10 == 0 {
            info!(
                "{:4} train loss: {:8.5} test loss: {:8.5}",
                i,
                ((2. - loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?) / 2.)
                    .acos()
                    .to_degrees(),
                ((2. - model.test_eval()?) / 2.).acos().to_degrees()
            );
        }
        if i % 100 == 0 {
            let loss = model
                .loss()?
                .to_dtype(candle_core::DType::F32)?
                .to_scalar::<f32>()?;
            if loss < initial_loss {
                varmap.save("weights.st")?;
                warn!("New saved loss: {}", loss);
                initial_loss = loss;
            }
        }
        if i % 1000 == 0 {
            let lr = optimiser.learning_rate();
            optimiser.set_learning_rate(0.5 * lr);
        }
    }
    let loss = model
        .loss()?
        .to_dtype(candle_core::DType::F32)?
        .to_scalar::<f32>()?;
    println!("Final loss {loss}");

    if loss < initial_loss {
        varmap.save("weights.st")?;
    }

    Ok(())
}
