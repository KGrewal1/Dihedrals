use parse_dihedrals::Dihedrals;
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

fn main() {
    let pathsample = Path::new("PATHSAMPLE");
    let systems: Dihedrals = fs::read_to_string(pathsample.join("min.dihedrals"))
        .expect("file not found")
        .parse()
        .unwrap();

    let mut file = File::create("min_dihedrals.csv").unwrap();
    for dihedral in systems {
        let angles = dihedral
            .dihedrals
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", angles).unwrap();
    }

    let systems: Dihedrals = fs::read_to_string(pathsample.join("ts.dihedrals"))
        .expect("file not found")
        .parse()
        .unwrap();

    let mut file = File::create("ts_dihedrals.csv").unwrap();
    for dihedral in systems {
        let angles = dihedral
            .dihedrals
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", angles).unwrap();
    }
}
