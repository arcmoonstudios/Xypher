/* build.rs */
/*▫~•◦────────────────────────────────────────────────────────────────────────────────────‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///◦────────────────────────────────────────────────────────────────────────────────────‣

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    fs,
};

type BuildResult<T> = Result<T, Box<dyn std::error::Error>>;

/// CUDA compilation configuration
struct CudaConfig {
    nvcc_path: PathBuf,
    cuda_root: PathBuf,
    compute_capabilities: Vec<String>,
    include_dirs: Vec<PathBuf>,
    library_dirs: Vec<PathBuf>,
    libraries: Vec<String>,
    optimization_level: String,
}

impl CudaConfig {
    fn detect() -> BuildResult<Self> {
        // Detect CUDA installation
        let cuda_root = Self::find_cuda_root()?;
        let nvcc_path = cuda_root.join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
        
        if !nvcc_path.exists() {
            return Err("NVCC not found in CUDA installation".into());
        }
        
        // Detect GPU compute capabilities
        let compute_capabilities = Self::detect_gpu_capabilities(&nvcc_path)?;
        
        // Setup paths
        let include_dirs = vec![
            cuda_root.join("include"),
            cuda_root.join("targets").join("x86_64-linux").join("include"), // Linux
            cuda_root.join("include").join("crt"),                            // Windows
        ];
        
        let library_dirs = vec![
            cuda_root.join("lib64"),                                          // Linux
            cuda_root.join("targets").join("x86_64-linux").join("lib"),      // Linux
            cuda_root.join("lib").join("x64"),                               // Windows
        ];
        
        let libraries = vec![
            "cuda".to_string(),
            "cudart".to_string(),
            "cublas".to_string(),
            "cublasLt".to_string(),
            "cudnn".to_string(),
            "curand".to_string(),
            "cufft".to_string(),
            "cusparse".to_string(),
            "cusolver".to_string(),
            "nccl".to_string(),
        ];
        
        let optimization_level = if cfg!(debug_assertions) { "O0".to_string() } else { "O3".to_string() };
        
        Ok(CudaConfig {
            nvcc_path,
            cuda_root,
            compute_capabilities,
            include_dirs,
            library_dirs,
            libraries,
            optimization_level,
        })
    }
    
    fn find_cuda_root() -> BuildResult<PathBuf> {
        // Try environment variable first
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let path = PathBuf::from(cuda_path);
            if path.exists() {
                return Ok(path);
            }
        }
        
        if let Ok(cuda_home) = env::var("CUDA_HOME") {
            let path = PathBuf::from(cuda_home);
            if path.exists() {
                return Ok(path);
            }
        }
        
        // Try standard installation paths
        let standard_paths = if cfg!(windows) {
            vec![
                PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"),
                PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"),
                PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"),
                PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3"),
            ]
        } else {
            vec![
                PathBuf::from("/usr/local/cuda"),
                PathBuf::from("/opt/cuda"),
                PathBuf::from("/usr/local/cuda-12.6"),
                PathBuf::from("/usr/local/cuda-12.5"),
                PathBuf::from("/usr/local/cuda-12.4"),
            ]
        };
        
        for path in standard_paths {
            if path.exists() {
                return Ok(path);
            }
        }
        
        Err("CUDA installation not found. Please set CUDA_PATH environment variable.".into())
    }
    
    fn detect_gpu_capabilities(nvcc_path: &Path) -> BuildResult<Vec<String>> {
        // Try to detect GPU capabilities automatically
        let output = Command::new(nvcc_path)
            .args(["--help"])
            .output();
        
        match output {
            Ok(_) => {
                // For production, support RTX 4080 (8.9) and common architectures
                Ok(vec![
                    "sm_75".to_string(),  // RTX 2080, Tesla T4
                    "sm_80".to_string(),  // RTX 3080, A100
                    "sm_86".to_string(),  // RTX 3090, RTX 4090
                    "sm_89".to_string(),  // RTX 4080, RTX 4070
                    "sm_90".to_string(),  // H100
                ])
            },
            Err(_) => {
                eprintln!("Warning: Could not detect GPU capabilities, using default set");
                Ok(vec!["sm_75".to_string(), "sm_80".to_string(), "sm_86".to_string(), "sm_89".to_string()])
            }
        }
    }
}

/// Compile CUDA kernels with optimal settings
fn compile_cuda_kernels(config: &CudaConfig, out_dir: &Path) -> BuildResult<()> {
    let cuda_dir = Path::new("cuda");
    
    if !cuda_dir.exists() {
        return Err("CUDA source directory not found".into());
    }
    
    // Find all .cu files
    let cu_files: Vec<_> = fs::read_dir(cuda_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "cu" {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    
    if cu_files.is_empty() {
        return Err("No CUDA source files found".into());
    }
    
    println!("Found {} CUDA source files", cu_files.len());
    
    // Compile each .cu file to .o
    let mut object_files = Vec::new();
    
    for cu_file in &cu_files {
        let obj_name = cu_file.file_stem().unwrap().to_string_lossy().to_string() + ".o";
        let obj_path = out_dir.join(&obj_name);
        
        println!("Compiling {}", cu_file.display());
        
        let mut nvcc_cmd = Command::new(&config.nvcc_path);
        
        // Basic compilation flags
        nvcc_cmd
            .arg("-c")
            .arg(cu_file)
            .arg("-o")
            .arg(&obj_path);
        
        // Optimization level
        nvcc_cmd.arg(format!("-{}", config.optimization_level));
        
        // Compute capabilities (compile for multiple architectures)
        for capability in &config.compute_capabilities {
            nvcc_cmd.arg(format!("-gencode=arch=compute_{},code=sm_{}", 
                                &capability[3..], &capability[3..]));
        }
        
        // Enable tensor core operations
        nvcc_cmd.arg("-DWITH_TENSOR_CORES=1");
        
        // Include directories
        for include_dir in &config.include_dirs {
            if include_dir.exists() {
                nvcc_cmd.arg(format!("-I{}", include_dir.display()));
            }
        }
        
        // Compiler flags for performance
        nvcc_cmd
            .arg("-std=c++17")
            .arg("--use_fast_math")
            .arg("--restrict")
            .arg("-Xcompiler=-fPIC") // Position independent code for shared libraries
            .arg("-Xcompiler=-march=native") // Native CPU optimization
            .arg("-Xcompiler=-mtune=native");
        
        // Debug vs Release specific flags
        if cfg!(debug_assertions) {
            nvcc_cmd
                .arg("-g")
                .arg("-G")
                .arg("-lineinfo");
        } else {
            nvcc_cmd
                .arg("-DNDEBUG")
                .arg("--ptxas-options=-v"); // Verbose PTX assembly for optimization info
        }
        
        // Enable cooperative groups and other advanced features
        nvcc_cmd
            .arg("--extended-lambda")
            .arg("--expt-relaxed-constexpr")
            .arg("-rdc=true"); // Relocatable device code for multi-file projects
        
        // Execute compilation
        let output = nvcc_cmd.output()?;
        
        if !output.status.success() {
            eprintln!("NVCC stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("NVCC stderr: {}", String::from_utf8_lossy(&output.stderr));
            return Err(format!("Failed to compile {}", cu_file.display()).into());
        }
        
        if !obj_path.exists() {
            return Err(format!("Object file not created: {}", obj_path.display()).into());
        }
        
        object_files.push(obj_path);
        println!("Successfully compiled {}", cu_file.display());
    }
    
    // Compile C++ wrapper
    let cpp_files = vec!["cuda/xypher_wrapper.cpp"];
    
    for cpp_file in cpp_files {
        let cpp_path = Path::new(cpp_file);
        if !cpp_path.exists() {
            continue;
        }
        
        let obj_name = cpp_path.file_stem().unwrap().to_string_lossy().to_string() + "_cpp.o";
        let obj_path = out_dir.join(&obj_name);
        
        println!("Compiling {}", cpp_path.display());
        
        let mut nvcc_cmd = Command::new(&config.nvcc_path);
        nvcc_cmd
            .arg("-c")
            .arg(cpp_path)
            .arg("-o")
            .arg(&obj_path)
            .arg(format!("-{}", config.optimization_level))
            .arg("-std=c++17")
            .arg("-Xcompiler=-fPIC");
        
        // Include directories
        for include_dir in &config.include_dirs {
            if include_dir.exists() {
                nvcc_cmd.arg(format!("-I{}", include_dir.display()));
            }
        }
        
        let output = nvcc_cmd.output()?;
        
        if !output.status.success() {
            eprintln!("NVCC C++ stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("NVCC C++ stderr: {}", String::from_utf8_lossy(&output.stderr));
            return Err(format!("Failed to compile {}", cpp_path.display()).into());
        }
        
        object_files.push(obj_path);
        println!("Successfully compiled {}", cpp_path.display());
    }
    
    // Link all object files into static library
    link_objects_into_library(&object_files, out_dir, config)?;
    
    println!("Successfully created CUDA static library");
    Ok(())
}

/// Links object files into a static library.
fn link_objects_into_library(object_files: &[PathBuf], out_dir: &Path, config: &CudaConfig) -> BuildResult<()> {
    #[cfg(windows)]
    {
        let lib_name = "xypher_cuda.lib";
        let lib_path = out_dir.join(lib_name);
        println!("Creating static library {}", lib_path.display());

        // Prefer using lib.exe from Visual Studio for creating the static library
        if which::which("lib.exe").is_ok() {
            let mut lib_cmd = Command::new("lib.exe");
            lib_cmd.arg(format!("/OUT:{}", lib_path.display()));
            for obj_file in object_files {
                lib_cmd.arg(obj_file);
            }
            let output = lib_cmd.output()?;
            if !output.status.success() {
                eprintln!("LIB stdout: {}", String::from_utf8_lossy(&output.stdout));
                eprintln!("LIB stderr: {}", String::from_utf8_lossy(&output.stderr));
                return Err("Failed to create static library with lib.exe".into());
            }
        } else {
            // Fallback to nvcc
            let mut nvcc_cmd = Command::new(&config.nvcc_path);
            nvcc_cmd.arg("-lib");
            for obj_file in object_files {
                nvcc_cmd.arg(obj_file);
            }
            nvcc_cmd.arg("-o").arg(&lib_path);
            let output = nvcc_cmd.output()?;
            if !output.status.success() {
                eprintln!("NVCC static lib stdout: {}", String::from_utf8_lossy(&output.stdout));
                eprintln!("NVCC static lib stderr: {}", String::from_utf8_lossy(&output.stderr));
                return Err("Failed to create static library with NVCC".into());
            }
        }

        if !lib_path.exists() {
            return Err(format!("Static library not created: {}", lib_path.display()).into());
        }
        println!("Successfully created static library in OUT_DIR");
    }
    #[cfg(not(windows))]
    {
        let lib_name = "libxypher_cuda.a";
        let lib_path = out_dir.join(lib_name);
        println!("Creating static library {}", lib_path.display());
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("rcs").arg(&lib_path);
        for obj_file in object_files {
            ar_cmd.arg(obj_file);
        }
        let output = ar_cmd.output()?;
        if !output.status.success() {
            eprintln!("AR stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("AR stderr: {}", String::from_utf8_lossy(&output.stderr));
            return Err("Failed to create static library".into());
        }
    }
    Ok(())
}


/// Setup CPU-specific optimizations
fn setup_cpu_optimizations() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    match target_arch.as_str() {
        "x86_64" => {
            // Platform-specific optimizations
            match target_os.as_str() {
                "linux" => {
                    println!("cargo:rustc-link-arg=-Wl,--as-needed");
                },
                "windows" => {
                    println!("cargo:rustc-link-arg=/OPT:REF");
                    println!("cargo:rustc-link-arg=/OPT:ICF");
                },
                "macos" => {
                    println!("cargo:rustc-link-arg=-dead_strip");
                },
                _ => {}
            }
        },
        "aarch64" => {
            // ARM optimizations
            println!("cargo:rustc-cfg=target_feature=\"neon\"");
        },
        _ => {}
    }
}

/// Generate FFI bindings
fn generate_ffi_bindings() -> BuildResult<()> {
    let bindings = bindgen::Builder::default()
        .header("cuda/xypher_kernels.cu")
        .header("cuda/xypher_wrapper.cpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("cuda_.*")
        .allowlist_function("xypher_.*")
        .allowlist_type("CudaError")
        .allowlist_var("CUDA_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings
        .write_to_file(out_path.join("cuda_bindings.rs"))?;
    
    println!("Generated FFI bindings");
    Ok(())
}

/// Main build script
fn main() -> BuildResult<()> {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    
    // Setup CPU optimizations regardless of CUDA availability
    setup_cpu_optimizations();
    
    // Check if CUDA feature is enabled
    let cuda_enabled = cfg!(feature = "cuda");
    
    if cuda_enabled {
        println!("CUDA feature enabled, attempting CUDA compilation...");
        
        match CudaConfig::detect() {
            Ok(config) => {
                println!("Found CUDA installation at: {}", config.cuda_root.display());
                println!("NVCC path: {}", config.nvcc_path.display());
                println!("Compute capabilities: {:?}", config.compute_capabilities);
                
                let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

                // Compile CUDA kernels
                if let Err(e) = compile_cuda_kernels(&config, &out_dir) {
                    eprintln!("Warning: Failed to compile CUDA kernels: {e}");
                    eprintln!("Falling back to CPU-only mode");
                    println!("cargo:rustc-cfg=feature=\"cuda_unavailable\"");
                } else {
                    println!("CUDA compilation successful");
                    println!("cargo:rustc-cfg=feature=\"cuda_available\"");
                    
                    // Tell Rust linker about the library
                    println!("cargo:rustc-link-search=native={}", out_dir.display());
                    println!("cargo:rustc-link-lib=static=xypher_cuda");

                    // Link CUDA libraries
                     for lib_dir in &config.library_dirs {
                         if lib_dir.exists() {
                             println!("cargo:rustc-link-search=native={}", lib_dir.display());
                         }
                     }

                    for lib in &config.libraries {
                        println!("cargo:rustc-link-lib={}", lib);
                    }
                }
            },
            Err(e) => {
                eprintln!("Warning: CUDA not found: {e}");
                eprintln!("Building with CPU-only support");
                println!("cargo:rustc-cfg=feature=\"cuda_unavailable\"");
            }
        }
    } else {
        println!("CUDA feature disabled, building CPU-only");
        println!("cargo:rustc-cfg=feature=\"cuda_unavailable\"");
    }
    
    // Additional build configurations
    
    // Link math library on Unix
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=m");
    }
    
    // Windows-specific libraries
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=oleaut32");
        println!("cargo:rustc-link-lib=uuid");
    }

    let wasm_target = cfg!(feature = "wasm");
    if wasm_target {
        println!("WASM target enabled");
        // WASM-specific build configuration would go here
    }
    
    println!("Build configuration complete");
    Ok(())
}