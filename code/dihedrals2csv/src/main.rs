use anyhow::anyhow;
use parse_dihedrals::Dihedrals;
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

fn main() -> anyhow::Result<()> {
    let pathsample = Path::new("PATHSAMPLE");
    let systems = fs::read_to_string(pathsample.join("min.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?;

    let mut file = File::create("min_dihedrals.csv")?;
    for dihedral in systems {
        let angles = dihedral
            .dihedrals
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", angles)?;
    }

    let systems = fs::read_to_string(pathsample.join("ts.dihedrals"))?
        .parse::<Dihedrals>()
        .map_err(|err| anyhow!(err))?;

    let mut file = File::create("ts_dihedrals.csv")?;
    for dihedral in systems {
        let angles = dihedral
            .dihedrals
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", angles)?;
    }
    Ok(())
}
