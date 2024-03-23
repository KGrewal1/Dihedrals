#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::range_plus_one
)]
use anyhow::anyhow;
use candle_core::Tensor;
use candle_nn::Module;
use env_logger::Builder;
use log::LevelFilter;
use parse_dihedrals::{Dihedral, Dihedrals};
use plotters::{prelude::*, style::full_palette::DEEPPURPLE_500};
use rayon::prelude::*;
use std::{fs, path::Path};

// mod nn_arch;
mod setup_model;
const NDIHEDRALS: usize = 178;

fn main() -> anyhow::Result<()> {
    let mut builder = Builder::new();
    builder.format_target(false);
    builder.format_timestamp(None);
    builder.format_level(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();

    let pathsample = Path::new("PATHSAMPLE");

    let dev = candle_core::Device::cuda_if_available(0)?;
    let model = setup_model::setup_connection()?;
    let minima: Vec<Dihedral> = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?
        .into();

    let preds = minima
        .par_iter()
        .enumerate()
        .flat_map(|(i, min_1)| {
            minima.par_iter().skip(i + 1).map(|min_2| {
                let mut angles: Vec<f64> = Vec::with_capacity(2 * NDIHEDRALS);
                angles.extend(min_1.dihedrals.iter());
                angles.extend(min_2.dihedrals.iter());
                Tensor::from_slice(&angles, (1, 1, 2, NDIHEDRALS), &dev)
                    .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
                    .and_then(|input| model.forward(&input))
                    .and_then(|pred| pred.reshape(()))
                    .and_then(|pred| pred.to_scalar::<f32>())
            })
        })
        .collect::<Result<Vec<f32>, _>>()?;

    let n_items = preds.len() as f64;

    println!("n_items: {}", n_items);

    let nbuckets = 200u32;
    let root = BitMapBackend::new("hist.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(90)
        .y_label_area_size(110)
        .margin(30)
        .caption("Frequency of Predictions", ("sans-serif", 50.0))
        .build_cartesian_2d(0u32..nbuckets + 1, 0f64..0.8)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_labels(5)
        .x_label_formatter(&|x| format!("{}%", (*x as f32 / nbuckets as f32 * 100.) as u8))
        .y_label_formatter(&|y| format!("{:.0}%", (*y * 100.0) as u32))
        .y_desc("Frequency")
        .x_desc("Probability of Connection")
        .label_style(("sans-serif", 30))
        .axis_desc_style(("sans-serif", 40))
        .draw()?;

    let hist = Histogram::vertical(&chart)
        .style(DEEPPURPLE_500.filled())
        .margin(0)
        .data(
            preds
                .iter()
                .map(|x| ((x * nbuckets as f32).round() as u32, n_items.recip())),
        );

    chart.draw_series(hist)?;

    let root = BitMapBackend::new("hist2.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(90)
        .y_label_area_size(110)
        .margin(30)
        .caption("Frequency of Predictions", ("sans-serif", 50.0))
        .build_cartesian_2d(0u32..nbuckets + 1, 0f64..0.005)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_labels(5)
        .x_label_formatter(&|x| format!("{}%", (*x as f32 / nbuckets as f32 * 100.) as u8))
        .y_label_formatter(&|y| format!("{:.1}%", (*y * 100.0)))
        .y_desc("Frequency")
        .x_desc("Probability of Connection")
        .label_style(("sans-serif", 30))
        .axis_desc_style(("sans-serif", 40))
        .draw()?;

    let hist = Histogram::vertical(&chart)
        .style(DEEPPURPLE_500.filled())
        .margin(0)
        .data(
            preds
                .iter()
                .map(|x| ((x * nbuckets as f32).round() as u32, n_items.recip())),
        );

    chart.draw_series(hist)?;

    Ok(())
}
