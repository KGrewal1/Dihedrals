use anyhow::{anyhow, Context};
use candle_core::Tensor;
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
    let minima: Vec<Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let connections: BTreeMap<(usize, usize), usize> =
        fs::read_to_string(pathsample.join("ts.data"))?
            .parse::<TransitionStates>()
            .map_err(|err| anyhow!(err))?
            .into();

    let mut connected_mins: Vec<(usize, usize)> = Vec::with_capacity(2 * connections.len());
    let mut unconnected_mins: Vec<(usize, usize)> = Vec::with_capacity(minima.len() * minima.len());

    (1..=minima.len()).for_each(|i| {
        (i + 1..=minima.len()).for_each(|j| {
            let cx = connections.get(&(i, j));
            if cx.is_some() {
                connected_mins.push((i, j));
                connected_mins.push((j, i));
            } else {
                unconnected_mins.push((i, j));
                unconnected_mins.push((j, i));
            }
        });
    });

    println!("Connected: {}", connected_mins.len());
    println!("Unconnected: {}", unconnected_mins.len());

    let ntrain_cx = 8000;
    let ntrain_ucx = 80_000;
    let ntest_cx = connected_mins.len() - ntrain_cx;
    let ntest_ucx = 250_000;

    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(42);
    // shuffle the orders for randomness
    connected_mins.shuffle(&mut rng);
    unconnected_mins.shuffle(&mut rng);

    let (cx_train, cx_test) = connected_mins.split_at(ntrain_cx);
    let (ucx_data, _) = unconnected_mins.split_at(ntrain_ucx + ntest_ucx);
    let (ucx_train, ucx_test) = ucx_data.split_at(ntrain_ucx);
    println!("cx_train: {}", cx_train.len());
    println!("cx_test: {}", cx_test.len());
    println!("ucx_train: {}", ucx_train.len());
    println!("ucx_test: {}", ucx_test.len());

    let minima: BTreeMap<usize, Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let mut connected_mins_train: Vec<f64> =
        Vec::with_capacity((ntrain_ucx + ntrain_cx) * 2 * NDIHEDRALS);

    for (i, j) in cx_train {
        connected_mins_train.extend(minima.get(i).context("minima missing")?.dihedrals.iter());
        connected_mins_train.extend(minima.get(j).context("minima missing")?.dihedrals.iter());
    }

    let mut connected_mins_test: Vec<f64> = Vec::with_capacity(ntest_cx * 2 * NDIHEDRALS);

    for (i, j) in cx_test {
        connected_mins_test.extend(minima.get(i).context("minima missing")?.dihedrals.iter());
        connected_mins_test.extend(minima.get(j).context("minima missing")?.dihedrals.iter());
    }

    let mut unconnected_mins_train: Vec<f64> = Vec::with_capacity(ntrain_ucx * 2 * NDIHEDRALS);

    for (i, j) in ucx_train {
        unconnected_mins_train.extend(minima.get(i).context("minima missing")?.dihedrals.iter());
        unconnected_mins_train.extend(minima.get(j).context("minima missing")?.dihedrals.iter());
    }

    let mut unconnected_mins_test: Vec<f64> = Vec::with_capacity(ntest_ucx * 2 * NDIHEDRALS);

    for (i, j) in ucx_test {
        unconnected_mins_test.extend(minima.get(i).context("minima missing")?.dihedrals.iter());
        unconnected_mins_test.extend(minima.get(j).context("minima missing")?.dihedrals.iter());
    }

    connected_mins_train.extend(unconnected_mins_train.iter());

    let mut train_vals: Vec<f64> = Vec::with_capacity(ntrain_cx + ntrain_ucx);
    train_vals.resize(ntrain_cx, 1.0);
    train_vals.resize(ntrain_cx + ntrain_ucx, 0.0);

    let ratio = ntrain_ucx as f64 / ntrain_cx as f64;
    let mut train_weights: Vec<f64> = Vec::with_capacity(ntrain_cx + ntrain_ucx);
    train_weights.resize(ntrain_cx, ratio);
    train_weights.resize(ntrain_cx + ntrain_ucx, 1.0);

    let mut test_vals_cx: Vec<f64> = Vec::with_capacity(ntest_cx);
    test_vals_cx.resize(ntest_cx, 1.0);

    let mut test_vals_ucx: Vec<f64> = Vec::with_capacity(ntest_ucx);
    test_vals_ucx.resize(ntest_ucx, 0.0);

    let train_input = Tensor::from_slice(
        &connected_mins_train,
        (ntrain_cx + ntrain_ucx, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let test_input_cx = Tensor::from_slice(
        &connected_mins_test,
        (ntest_cx, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let test_input_ucx = Tensor::from_slice(
        &unconnected_mins_test,
        (ntest_ucx, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let train_output = Tensor::from_slice(
        &train_vals,
        (ntrain_cx + ntrain_ucx, 1),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let train_weights = Tensor::from_slice(
        &train_weights,
        (ntrain_cx + ntrain_ucx, 1),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let test_output_cx =
        Tensor::from_slice(&test_vals_cx, (ntest_cx, 1), &candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;

    let test_output_ucx =
        Tensor::from_slice(&test_vals_ucx, (ntest_ucx, 1), &candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;

    candle_core::safetensors::save(
        &HashMap::from([
            ("traininput", train_input),
            ("trainoutput", train_output),
            ("trainweights", train_weights),
            ("testinputcx", test_input_cx),
            ("testoutputcx", test_output_cx),
            ("testinputucx", test_input_ucx),
            ("testoutputucx", test_output_ucx),
        ]),
        "dihedral_class_data.st",
    )?;

    println!("Done");

    Ok(())
}
