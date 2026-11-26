//! Which MKL copy did this process actually load? A stale MKL earlier on
//! `LD_LIBRARY_PATH` (or `PATH` on Windows) wins silently. Linux only here:
//! reads `/proc/self/maps`. Windows (`GetModuleFileNameW`) is not implemented.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};

fn loaded_mkl_dirs() -> Vec<PathBuf> {
    // Touch MKL so a lazily loaded library is mapped before we read the maps.
    let mut v = [0u8; 256];
    unsafe { onemkl_sys::MKL_Get_Version_String(v.as_mut_ptr().cast(), v.len() as i32) };
    eprintln!(
        "MKL VERSION: {}",
        String::from_utf8_lossy(&v).trim_end_matches(char::from(0))
    );
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let mut dirs: Vec<PathBuf> = maps
        .lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .filter(|p| {
            Path::new(p)
                .file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("libmkl"))
        })
        .filter_map(|p| Path::new(p).parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

fn check(expected_root: &Path, dirs: &[PathBuf]) -> Result<(), String> {
    match dirs.iter().find(|d| !d.starts_with(expected_root)) {
        Some(d) => Err(format!(
            "STALE MKL: loaded libs from {} but expected under {}",
            d.display(),
            expected_root.display()
        )),
        None => Ok(()),
    }
}

#[test]
fn loaded_mkl_comes_from_mklroot() {
    let dirs = loaded_mkl_dirs();
    eprintln!("LOADED MKL DIRS: {dirs:?}");
    if dirs.is_empty() {
        eprintln!("statically linked MKL: nothing to compare");
        return;
    }
    // Control: a deliberately wrong root must fail, so the check can fail.
    assert!(check(Path::new("/definitely/not/mkl"), &dirs).is_err());
    match std::env::var_os("MKLROOT") {
        Some(root) => {
            // /proc/self/maps shows resolved paths and MKLROOT/lib may be symlinks
            // (it is, on the WSL rig: pip mkl), so compare the resolved library dir.
            let resolved = Path::new(&root).join("lib").canonicalize().ok();
            let lib_real = std::fs::read_dir(Path::new(&root).join("lib"))
                .ok()
                .and_then(|mut d| {
                    d.find_map(|e| {
                        e.ok()?
                            .path()
                            .canonicalize()
                            .ok()?
                            .parent()
                            .map(Path::to_path_buf)
                    })
                });
            let expected = lib_real
                .or(resolved)
                .unwrap_or_else(|| PathBuf::from(&root));
            eprintln!(
                "EXPECTED (resolved from MKLROOT/lib): {}",
                expected.display()
            );
            check(&expected, &dirs).unwrap()
        }
        None => eprintln!("MKLROOT unset: only the control ran"),
    }
}
