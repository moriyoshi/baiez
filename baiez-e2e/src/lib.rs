// SPDX-License-Identifier: Apache-2.0
//! Monty scenarios can only reach the OS through the host verbs in `world`.

mod world;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use monty::{MontyRun, RunProgress};
use monty_types::{
    CompileOptions, ExtFunctionResult, MontyObject, NameLookupResult, PrintWriter, ResourceLimits,
    ResourceTracker,
};

use world::World;

#[derive(Debug)]
pub struct Outcome {
    pub error: Option<String>,
    pub output: String,
    pub calls: u64,
    pub elapsed: Duration,
}

pub fn scenarios() -> std::io::Result<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../e2e/scenarios");
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "py"))
        .collect();
    paths.sort();
    Ok(paths)
}

pub fn run_file(path: &Path, timeout: Duration) -> Outcome {
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |value| value.to_string_lossy().into_owned(),
    );
    let started = Instant::now();
    let mut output = String::new();
    let mut calls = 0;
    let result = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
        .and_then(|source| execute(&name, &source, timeout, &mut output, &mut calls));
    Outcome {
        error: result.err(),
        output,
        calls,
        elapsed: started.elapsed(),
    }
}

fn execute(
    name: &str,
    source: &str,
    timeout: Duration,
    output: &mut String,
    calls: &mut u64,
) -> Result<(), String> {
    if !source
        .lines()
        .any(|line| line.trim_start().starts_with("assert "))
    {
        return Err("scenario has no assert statement".to_owned());
    }
    let mut world = World::new(name, timeout).map_err(|error| error.to_string())?;
    let run = MontyRun::new(
        source.to_owned(),
        name,
        Vec::new(),
        CompileOptions::default(),
    )
    .map_err(|error| error.summary())?;
    let tracker = ResourceTracker::new(ResourceLimits::default().max_duration(timeout));
    let mut progress = run
        .start(Vec::new(), tracker, writer(output))
        .map_err(|error| error.summary())?;
    loop {
        progress = match progress {
            RunProgress::Complete(_) => break,
            RunProgress::NameLookup(lookup) => {
                let value = if World::is_verb(&lookup.name) {
                    NameLookupResult::Value(MontyObject::Function {
                        name: lookup.name.clone(),
                        docstring: None,
                    })
                } else {
                    NameLookupResult::Undefined
                };
                lookup
                    .resume(value, writer(output))
                    .map_err(|error| error.summary())?
            }
            RunProgress::FunctionCall(call) => {
                let result = if World::is_verb(&call.function_name) {
                    *calls += 1;
                    match world.call(&call.function_name, &call.args, &call.kwargs) {
                        Ok(value) => ExtFunctionResult::Return(value),
                        Err(error) => ExtFunctionResult::Error(error),
                    }
                } else {
                    ExtFunctionResult::NotFound(call.function_name.clone())
                };
                call.resume(result, writer(output))
                    .map_err(|error| error.summary())?
            }
            RunProgress::OsCall(_) => {
                return Err("scenario attempted an OS call outside its host verbs".to_owned());
            }
            RunProgress::ResolveFutures(_) => {
                return Err("scenario awaited an unsupported future".to_owned());
            }
        };
    }
    if *calls == 0 {
        return Err("scenario called no fixture verb".to_owned());
    }
    Ok(())
}

fn writer(output: &mut String) -> PrintWriter<'_> {
    PrintWriter::CollectString(output, Some(1 << 20))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenarios_compile_and_assert() {
        let paths = scenarios().expect("scenario directory");
        assert!(!paths.is_empty());
        for path in paths {
            let source = std::fs::read_to_string(&path).expect("source");
            assert!(
                source
                    .lines()
                    .any(|line| line.trim_start().starts_with("assert ")),
                "{}",
                path.display()
            );
            MontyRun::new(
                source,
                path.to_string_lossy().as_ref(),
                Vec::new(),
                CompileOptions::default(),
            )
            .unwrap_or_else(|error| panic!("{}: {}", path.display(), error.summary()));
        }
    }
}
