use anyhow::anyhow;
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

    let mut connected_mins: Vec<f64> = Vec::with_capacity(2 * NDIHEDRALS * connections.len());
    let mut unconnected_mins: Vec<f64> = Vec::with_capacity(
        minima.len() * minima.len() * NDIHEDRALS * 2 - connected_mins.capacity(),
    );

    // let raw = Vec::from_raw_parts();
    minima.iter().enumerate().for_each(|(i, min_1)| {
        minima.iter().skip(i + 1).for_each(|min_2| {
            let cx = connections.get(&(min_1.id, min_2.id));
            if cx.is_some() {
                connected_mins.extend(min_1.dihedrals.iter());
                connected_mins.extend(min_2.dihedrals.iter());
                connected_mins.extend(min_2.dihedrals.iter());
                connected_mins.extend(min_1.dihedrals.iter());
            } else {
                unconnected_mins.extend(min_1.dihedrals.iter());
                unconnected_mins.extend(min_2.dihedrals.iter());
                unconnected_mins.extend(min_2.dihedrals.iter());
                unconnected_mins.extend(min_1.dihedrals.iter());
            }
        });
    });

    let connected_mins: Vec<[f64; 2 * NDIHEDRALS]> =
        bytemuck::try_cast_vec(connected_mins).map_err(|err| err.0)?;

    let mut unconnected_mins: Vec<[f64; 2 * NDIHEDRALS]> =
        bytemuck::try_cast_vec(unconnected_mins).map_err(|err| err.0)?;

    println!("Connected: {}", connected_mins.len());
    println!("Unconnected: {}", unconnected_mins.len());

    let mut tt_mask: Vec<bool> = Vec::with_capacity(connected_mins.len());
    let ntrain = 8000;
    let ntest_cx = connected_mins.len() - ntrain;
    let ntest_ucx = 250_000;
    tt_mask.resize(ntrain, true);
    tt_mask.resize(connected_mins.len(), false);

    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(42);
    tt_mask.shuffle(&mut rng);
    let mut cx_train: Vec<[f64; 2 * NDIHEDRALS]> = Vec::with_capacity(ntrain);
    let mut cx_test: Vec<[f64; 2 * NDIHEDRALS]> = Vec::with_capacity(ntest_cx);

    for (min_pair, mask) in connected_mins.into_iter().zip(tt_mask.iter()) {
        if *mask {
            cx_train.push(min_pair);
        } else {
            cx_test.push(min_pair);
        }
    }

    tt_mask.shuffle(&mut rng);
    unconnected_mins.shuffle(&mut rng);
    let ucx_test: Vec<[f64; 2 * NDIHEDRALS]> = unconnected_mins.drain(0..ntest_ucx).collect();
    let mut ucx_train: Vec<[f64; 2 * NDIHEDRALS]> = Vec::with_capacity(ntrain);

    for (min_pair, mask) in unconnected_mins.into_iter().zip(tt_mask.iter()) {
        if *mask {
            ucx_train.push(min_pair);
        }
    }

    let mut cx_train: Vec<f64> = bytemuck::try_cast_vec(cx_train).map_err(|err| err.0)?;

    let cx_test: Vec<f64> = bytemuck::try_cast_vec(cx_test).map_err(|err| err.0)?;

    let ucx_train: Vec<f64> = bytemuck::try_cast_vec(ucx_train).map_err(|err| err.0)?;

    let ucx_test: Vec<f64> = bytemuck::try_cast_vec(ucx_test).map_err(|err| err.0)?;

    cx_train.extend(ucx_train.iter());

    let mut train_vals: Vec<f64> = Vec::with_capacity(2 * ntrain);
    train_vals.resize(ntrain, 1.0);
    train_vals.resize(2 * ntrain, 0.0);

    let mut test_vals_cx: Vec<f64> = Vec::with_capacity(ntest_cx);
    test_vals_cx.resize(ntest_cx, 1.0);

    let mut test_vals_ucx: Vec<f64> = Vec::with_capacity(ntest_ucx);
    test_vals_ucx.resize(ntest_ucx, 0.0);

    let train_input = Tensor::from_slice(
        &cx_train,
        (2 * ntrain, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let test_input_cx = Tensor::from_slice(
        &cx_test,
        (ntest_cx, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let test_input_ucx = Tensor::from_slice(
        &ucx_test,
        (ntest_ucx, 1, 2, NDIHEDRALS),
        &candle_core::Device::Cpu,
    )?
    .to_dtype(candle_core::DType::F32)?;

    let train_output = Tensor::from_slice(&train_vals, (2 * ntrain, 1), &candle_core::Device::Cpu)?
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
