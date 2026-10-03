// SPDX-License-Identifier: Apache-2.0
//! Optional dynamic LightGBM importer. No link-time dependency on LightGBM.

use std::error::Error;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

use libloading::Library;

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*const c_char, *mut c_int, *mut Handle) -> c_int;
type Dump = unsafe extern "C" fn(Handle, c_int, c_int, c_int, i64, *mut i64, *mut c_char) -> c_int;
type Free = unsafe extern "C" fn(Handle) -> c_int;
type LastError = unsafe extern "C" fn() -> *const c_char;

pub fn dump_model(path: &Path, library_path: Option<&str>) -> Result<String, Box<dyn Error>> {
    let candidates: Vec<&str> = match library_path {
        Some(path) => vec![path],
        None => vec!["lib_lightgbm.so", "lib_lightgbm.dylib", "lightgbm.dll"],
    };
    let mut last_error = None;
    for candidate in candidates {
        // SAFETY: the library is held alive until all its function pointers and
        // the booster handle have finished being used below.
        let lib = match unsafe { Library::new(candidate) } {
            Ok(lib) => lib,
            Err(error) => {
                last_error = Some(error.to_string());
                continue;
            }
        };
        // SAFETY: these signatures match LightGBM's public C API.
        unsafe {
            let create: Create = *lib.get(b"LGBM_BoosterCreateFromModelfile\0")?;
            let dump: Dump = *lib.get(b"LGBM_BoosterDumpModel\0")?;
            let free: Free = *lib.get(b"LGBM_BoosterFree\0")?;
            let last_error: LastError = *lib.get(b"LGBM_GetLastError\0")?;
            let path = CString::new(path.to_str().ok_or("model path is not UTF-8")?)?;
            let mut handle = std::ptr::null_mut();
            let mut iterations = 0;
            if create(path.as_ptr(), &mut iterations, &mut handle) != 0 {
                return Err(CStr::from_ptr(last_error())
                    .to_string_lossy()
                    .into_owned()
                    .into());
            }
            let result = (|| {
                let mut buffer = vec![0_u8; 1024 * 1024];
                loop {
                    let mut needed = 0_i64;
                    let status = dump(
                        handle,
                        0,
                        -1,
                        0,
                        buffer.len() as i64,
                        &mut needed,
                        buffer.as_mut_ptr().cast(),
                    );
                    if status != 0 {
                        return Err(CStr::from_ptr(last_error())
                            .to_string_lossy()
                            .into_owned()
                            .into());
                    }
                    let needed = usize::try_from(needed)?;
                    if needed > buffer.len() {
                        buffer.resize(needed, 0);
                        continue;
                    }
                    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(needed);
                    return Ok(String::from_utf8(buffer[..end].to_vec())?);
                }
            })();
            free(handle);
            return result;
        }
    }
    Err(format!(
        "LightGBM shared library not found; pass --lib=PATH or set LD_LIBRARY_PATH ({})",
        last_error.unwrap_or_else(|| "no candidate found".into())
    )
    .into())
}
