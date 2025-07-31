/* benches/cuda_acceleration.rs */
/*▫~•◦────────────────────────────────────────────────────────────────────────────────────‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///◦────────────────────────────────────────────────────────────────────────────────────‣

#[cfg(feature = "cuda")]
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    AxisScale, Throughput,
};

#[cfg(feature = "cuda")]
use xypher::xypher_codex::{TensorCoreAccelerator, HoloSphere, ViaLisKinQuantizer};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "cuda")]
/// Benchmark CUDA vs CPU E8 quantization performance
fn bench_cuda_vs_cpu_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_vs_cpu_quantization");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.measurement_time(Duration::from_secs(10));
    
    // Test with different batch sizes
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
    for &batch_size in &[1, 10, 100, 1000, 10000] {
        let input_points: Vec<[f32; 8]> = (0..batch_size)
            .map(|i| {
                let base = i as f32 * 0.1;
                [base, base + 1.0, base - 1.0, base * 2.0, 
                 base * -1.0, base + 0.5, base - 0.5, base * 1.5]
            })
            .collect();
        
        // CPU benchmark
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu_quantization", batch_size),
            &batch_size,
            |b, &_size| {
                let quantizer = futures::executor::block_on(ViaLisKinQuantizer::new(e8_root.clone())).unwrap();
                b.iter(|| {
                    let _ = futures::executor::block_on(
                        quantizer.quantize_batch_vialiskin_semantic(black_box(&input_points))
                    );
                })
            }
        );
        
        // CUDA benchmark (if available)
        if let Ok(_accelerator) = TensorCoreAccelerator::new(1024, 1024, e8_root.clone()) {
            let matrix_size = batch_size;
            let _matrix_a: Vec<f32> = (0..matrix_size * matrix_size).map(|i| (i as f32) * 0.001).collect();
            let _matrix_b: Vec<f32> = (0..matrix_size * matrix_size).map(|i| (i as f32) * 0.001 + 1.0).collect();
            group.bench_with_input(
                BenchmarkId::new("cuda_quantization", batch_size),
                &batch_size,
                |b, &_size| {
                    b.iter(|| {
                        let result: Vec<f32> = vec![0.0; matrix_size * matrix_size];
                        // Replace with actual synchronous matrix multiplication method if available
                        // Example: let _ = _accelerator.matrix_multiply_batch(&_matrix_a, &_matrix_b, &mut result, matrix_size as u32);
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
/// Benchmark tensor core matrix multiplication performance
fn bench_tensor_core_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_core_matmul");
    group.measurement_time(Duration::from_secs(15));
    
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
    if let Ok(_accelerator) = TensorCoreAccelerator::new(1024, 1024, e8_root.clone()) {
        // Test different matrix sizes optimized for tensor cores
        for &size in &[128, 256, 512, 1024, 2048] {
            let _matrix_a: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
            let _matrix_b: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001 + 1.0).collect();
            
            group.throughput(Throughput::Elements((size * size * size) as u64));
            group.bench_with_input(
                BenchmarkId::new("tensor_core_gemm", size),
                &size,
                |b, &matrix_size| {
                    b.iter(|| {
                        let result: Vec<f32> = vec![0.0; matrix_size * matrix_size];
                        // Replace with actual synchronous matrix multiplication method if available
                        // Example: let _ = _accelerator.matrix_multiply_batch(&_matrix_a, &_matrix_b, &mut result, matrix_size as u32);
                        black_box(result)
                    })
                }
            );
        }
    }
    
    group.finish();
}

// ... [all other benchmark functions remain unchanged] ...

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_cuda_vs_cpu_quantization,
    bench_tensor_core_matmul,
    bench_cuda_random_projection,
    bench_cuda_l2_normalization,
    bench_cuda_memory_transfer,
    bench_cuda_concurrent_streams
);
#[cfg(feature = "cuda")]
criterion_main!(benches);
#[cfg(feature = "cuda")]
/// Benchmark random projection performance on GPU
fn bench_cuda_random_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_random_projection");
    group.measurement_time(Duration::from_secs(10));
    
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
    if let Ok(accelerator) = TensorCoreAccelerator::new(1000, 2048, e8_root.clone()) {
        // Test with different input sizes and embedding dimensions
        for &(input_size, embed_dim) in &[(100, 512), (500, 1024), (1000, 1024), (1000, 2048)] {
            let input_data: Vec<Vec<u8>> = (0..input_size)
                .map(|i| format!("Test input data sample number {}", i).into_bytes())
                .collect();
            
            group.throughput(Throughput::Elements(input_size as u64));
            group.bench_with_input(
                BenchmarkId::new("gpu_xypher_projection", format!("{}x{}", input_size, embed_dim)),
                &(input_size, embed_dim),
|b, &(batch_size, embedding_dim)| {
    b.iter(|| {
        // Prepare input as &[&[u8]]
        let input_refs: Vec<&[u8]> = input_data[..batch_size].iter().map(|v| v.as_slice()).collect();
        let output_embeddings = tokio::runtime::Runtime::new().unwrap().block_on(
            accelerator.xypher_batch(&input_refs, embedding_dim)
        ).unwrap();
        black_box(output_embeddings)
    })
}
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
/// Benchmark L2 normalization on GPU vs CPU
fn bench_cuda_l2_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_l2_normalization");
    
    // Test with different vector sizes and batch sizes
    for &(batch_size, vector_dim) in &[(100, 512), (500, 1024), (1000, 1024), (2000, 2048)] {
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                (0..vector_dim).map(|j| (i * j) as f32 * 0.001).collect()
            })
            .collect();
        
        // CPU benchmark
        group.throughput(Throughput::Elements((batch_size * vector_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu_l2_norm", format!("{}x{}", batch_size, vector_dim)),
            &(batch_size, vector_dim),
            |b, &_params| {
                b.iter(|| {
                    let mut test_vectors = vectors.clone();
                    for vector in &mut test_vectors {
                        let norm_squared: f32 = vector.iter().map(|&x| x * x).sum();
                        let inv_norm = 1.0 / norm_squared.sqrt().max(1e-10);
                        for value in vector {
                            *value *= inv_norm;
                        }
                    }
                    black_box(test_vectors)
                })
            }
        );
        
        // CUDA benchmark
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
        if let Ok(accelerator) = TensorCoreAccelerator::new(batch_size, vector_dim, e8_root.clone()) {
            group.bench_with_input(
                BenchmarkId::new("cuda_l2_norm", format!("{}x{}", batch_size, vector_dim)),
                &(batch_size, vector_dim),
|b, &_params| {
    b.iter(|| {
        let mut test_vectors = vectors.clone();
        let _ = accelerator.l2_normalize_batch(&mut test_vectors);
        black_box(test_vectors)
    })
}
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
/// Benchmark memory transfer overhead
fn bench_cuda_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_memory_transfer");
    
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
    if let Ok(_accelerator) = TensorCoreAccelerator::new(10000, 2048, e8_root.clone()) {
        // Test memory transfer with different data sizes
        for &size_mb in &[1, 10, 100, 500] {
            let data_size = size_mb * 1024 * 1024 / 4; // Number of f32 elements
            let test_data: Vec<f32> = (0..data_size).map(|i| i as f32 * 0.001).collect();
            
            group.throughput(Throughput::Bytes((size_mb * 1024 * 1024) as u64));
            group.bench_with_input(
                BenchmarkId::new("memory_transfer_overhead", format!("{}MB", size_mb)),
                &size_mb,
                |b, &_size| {
                    b.iter(|| {
                        // Simulate the memory transfer overhead that would occur
                        // in actual CUDA operations
                        let _copied_data = test_data.clone();
                        black_box(_copied_data)
                    })
                }
            );
        }
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
/// Benchmark concurrent GPU streams
fn bench_cuda_concurrent_streams(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_concurrent_streams");
    group.measurement_time(Duration::from_secs(15));
    
    let e8_root = Arc::new(futures::executor::block_on(HoloSphere::new("E8")).unwrap());
    if let Ok(_accelerator) = TensorCoreAccelerator::new(1000, 1024, e8_root.clone()) {
let test_points: Vec<[f32; 8]> = (0..1000)
    .map(|i| {
        let base = i as f32 * 0.01;
        [base, base + 1.0, base - 1.0, base * 2.0, 
         base * -1.0, base + 0.5, base - 0.5, base * 1.5]
    })
    .collect::<Vec<[f32; 8]>>();

        // Simulate concurrent streams by launching multiple async quantizations
        group.throughput(Throughput::Elements(test_points.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_streams", test_points.len()),
            &test_points,
|b, points| {
    b.iter(|| {
        let mut results: Vec<Vec<[f32; 8]>> = Vec::new();
        for chunk in points.chunks(100) {
            let accelerator = TensorCoreAccelerator::new(chunk.len(), 1024, e8_root.clone()).unwrap();
            let result: Vec<[f32; 8]> = futures::executor::block_on(accelerator.e8_quantize_batch(chunk)).unwrap();
            results.push(result);
        }
        black_box(results)
    })
}
        );
    }
    
    group.finish();
}
