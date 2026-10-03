// SPDX-License-Identifier: Apache-2.0
//! Run checked-in Monty scenarios against real yesnod and baiez processes.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

fn main() -> ExitCode {
    let mut paths = Vec::new();
    let mut timeout = 180u64;
    let mut show_output = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => {
                for path in baiez_e2e::scenarios().expect("scenario directory") {
                    println!("{}", path.file_stem().expect("stem").to_string_lossy());
                }
                return ExitCode::SUCCESS;
            }
            "--show-output" => show_output = true,
            "--timeout" => {
                timeout = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                if timeout == 0 {
                    eprintln!("--timeout needs a positive whole number of seconds");
                    return ExitCode::FAILURE;
                }
            }
            "--help" | "-h" => {
                println!("baiez-e2e [--list] [--show-output] [--timeout SECS] [SCENARIO ...]");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown option: {other}");
                return ExitCode::FAILURE;
            }
            other => paths.push(PathBuf::from(other)),
        }
    }
    let available = baiez_e2e::scenarios().expect("scenario directory");
    if paths.is_empty() {
        paths = available;
    } else {
        paths = paths
            .into_iter()
            .map(|path| {
                if path.exists() {
                    path
                } else {
                    available
                        .iter()
                        .find(|item| item.file_stem() == path.file_stem())
                        .cloned()
                        .unwrap_or(path)
                }
            })
            .collect();
    }
    let mut failed = 0;
    for path in &paths {
        let result = baiez_e2e::run_file(path, Duration::from_secs(timeout));
        let name = path.file_stem().expect("scenario name").to_string_lossy();
        if let Some(error) = result.error {
            failed += 1;
            println!(
                "FAIL {name} ({:.2}s, {} calls)",
                result.elapsed.as_secs_f64(),
                result.calls
            );
            if !result.output.is_empty() {
                print!("{}", result.output);
            }
            eprintln!("{error}");
        } else {
            println!(
                "ok   {name} ({:.2}s, {} calls)",
                result.elapsed.as_secs_f64(),
                result.calls
            );
            if show_output {
                print!("{}", result.output);
            }
        }
    }
    println!(
        "{} scenario(s): {} passed, {failed} failed",
        paths.len(),
        paths.len() - failed
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
