use anyhow::{anyhow, Context};
use candle_core::{Result, Tensor};
use parse_dihedrals::{Dihedral, Dihedrals};
use parse_tsdata::TransitionStates;
use rand::{seq::SliceRandom, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

const NDIHEDRALS: usize = 178;
fn main() -> anyhow::Result<()> {
    let pathsample = Path::new("PATHSAMPLE");

    let minima: BTreeMap<usize, Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let transition_states: BTreeMap<usize, Dihedral> =
        fs::read_to_string(pathsample.join("ts.dihedrals"))?
            .parse::<Dihedrals>()
            .map_err(|err| anyhow!(err))?
            .into();

    let connections = fs::read_to_string(pathsample.join("ts.data"))?
        .parse::<TransitionStates>()
        .map_err(|err| anyhow!(err))?;

    let mut train_input: Vec<f64> = Vec::new();
    let mut train_output: Vec<f64> = Vec::new();
    let mut test_input: Vec<f64> = Vec::new();
    let mut test_output: Vec<f64> = Vec::new();

    let mut tt_mask: Vec<bool> = Vec::with_capacity(connections.len());
    let ntrain = 4000;
    let ntest = connections.len() - ntrain;
    tt_mask.resize(ntrain, true);
    tt_mask.resize(connections.len(), false);

    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(42);
    tt_mask.shuffle(&mut rng);
    for (connection, mask) in connections.into_iter().zip(tt_mask) {
        let (ts, min_1, min_2) = (connection.ts, connection.min1, connection.min2);
        let ts = transition_states.get(&ts).context("ts not found")?;
        if mask {
            train_output.extend(ts.dihedrals.iter());
        } else {
            test_output.extend(ts.dihedrals.iter());
        }
        let min_1_val = minima.get(&min_1).context("min 1 not found")?;
        let min_2_val = minima.get(&min_2).context("min 2 not found")?;
        if mask {
            train_input.extend(min_1_val.dihedrals.iter());
            train_input.extend(min_2_val.dihedrals.iter());
        } else {
            test_input.extend(min_1_val.dihedrals.iter());
            test_input.extend(min_2_val.dihedrals.iter());
        }
    }

    let train_input = Tensor::from_slice(
        &train_input,
        (ntrain, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?;

    let test_input = Tensor::from_slice(
        &test_input,
        (ntest, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?;

    let train_output = Tensor::from_slice(
        &train_output,
        (ntrain, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?;

    let train_mean = &train_input.transpose(2, 3)?.mean(3)?.flatten_from(1)?;
    let mean_mse: f64 = angle_mse(&train_output, train_mean)?.to_scalar()?;
    println!("{}", ((2. - mean_mse) / 2.).acos().to_degrees());
    let test_output =
        Tensor::from_slice(&test_output, (ntest, NDIHEDRALS), &candle_core::Device::Cpu)?;

    candle_core::safetensors::save(
        &HashMap::from([
            ("traininput", train_input),
            ("trainoutput", train_output),
            ("testinput", test_input),
            ("testoutput", test_output),
        ]),
        "dihedral_data.st",
    )?;
    Ok(())
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
