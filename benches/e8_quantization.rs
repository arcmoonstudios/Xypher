/* benches/e8_quantization.rs */
/*▫~•◦────────────────────────────────────────────────────────────────────────────────────‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///◦────────────────────────────────────────────────────────────────────────────────────‣

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    AxisScale, Throughput,
};
use xypher::xypher_codex::{XypherCodex, ViaLisKinQuantizer, HoloSphere};
use std::sync::Arc;
use futures::executor;
use std::time::Duration;

/// Benchmark E8 lattice quantization performance across different scenarios
fn bench_e8_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("e8_quantization");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    let holosphere = Arc::new(executor::block_on(HoloSphere::new("http://xypher.arcmoon.ai/semantic/")).unwrap());
    let quantizer: ViaLisKinQuantizer = executor::block_on(ViaLisKinQuantizer::new(holosphere.clone())).unwrap();
    
    // Benchmark single point quantization
    group.bench_function("single_point", |b| {
        let point = [1.5, 2.3, -0.7, 3.1, -1.2, 0.8, 2.7, -2.1];
        b.iter(|| {
            black_box(quantizer.quantize_e8_point(black_box(&point)))
        })
    });
    
    // Benchmark batch quantization with varying sizes
    for &batch_size in &[1, 10, 100, 1000, 10000] {
        let input_points: Vec<[f32; 8]> = (0..batch_size)
            .map(|i| {
                let base = i as f32 * 0.1;
                [base, base + 1.0, base - 1.0, base * 2.0, 
                 base * -1.0, base + 0.5, base - 0.5, base * 1.5]
            })
            .collect();
        
        // Removed unused output_points variable
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_quantization", batch_size),
            &batch_size,
            |b, &_size| {
                b.iter(|| {
                    // Use the async batch quantization method and block on it
                    let _ = futures::executor::block_on(
                        quantizer.quantize_batch_vialiskin_semantic(black_box(&input_points))
                    );
                })
            }
        );
    }
    
    // Benchmark Conway construction components
    group.bench_function("integer_lattice_only", |b| {
        let point = [1.7, 2.3, -0.9, 3.4, -1.8, 0.6, 2.9, -2.4];
        b.iter(|| {
            // This would benchmark just the integer lattice part
            // For now, use the full quantization as a proxy
            black_box(quantizer.quantize_e8_point(black_box(&point)))
        })
    });
    
    // Benchmark distance computation
    group.bench_function("distance_computation", |b| {
        let point_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let point_b = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1];
        b.iter(|| {
            // Manual distance computation for benchmarking
            let mut sum = 0.0f32;
            for i in 0..8 {
                let diff = point_a[i] - point_b[i];
                sum += diff * diff;
            }
            black_box(sum)
        })
    });
    
    group.finish();
}

/// Benchmark E8 tokenless encoder performance
fn bench_tokenless_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenless_encoder");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Test with different encoder configurations
    for &blocks in &[16, 64, 128, 256] {
        let embedding_dim = blocks * 8;
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_text_encoding", embedding_dim),
            &embedding_dim,
            |b, &_dim| {
                let text = "The quick brown fox jumps over the lazy dog. \
                           This is a longer text sample for testing tokenless encoding performance.";
                b.iter(|| {
                    let encoder = futures::executor::block_on(XypherCodex::new(blocks, 42)).unwrap();
                    black_box(futures::executor::block_on(encoder.encode_text(black_box(text))))
                })
            }
        );
    }
    
    // Benchmark with different text lengths
    for &text_length in &[10, 100, 1000, 10000] {
        let text: String = "a".repeat(text_length);
        group.throughput(Throughput::Bytes(text_length as u64));
        group.bench_with_input(
            BenchmarkId::new("variable_length_text", text_length),
            &text_length,
            |b, &_len| {
                b.iter(|| {
                    let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
                    black_box(futures::executor::block_on(encoder.encode_text(black_box(&text))))
                })
            }
        );
    }
    
    // Benchmark batch encoding
    let test_texts = vec![
        "Short text",
        "Medium length text with more words and complexity",
        "This is a much longer text sample that contains multiple sentences. \
         It includes various punctuation marks, numbers like 123 and 456, \
         and tests the encoder's ability to handle diverse content types.",
        "Code example: fn main() { println!(\"Hello, world!\"); }",
        "Mathematical expression: f(x) = x² + 2x + 1",
    ];
    
    group.throughput(Throughput::Elements(test_texts.len() as u64));
    group.bench_function("batch_encoding", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            let batch_bytes: Vec<Vec<u8>> = test_texts.iter().map(|s| s.as_bytes().to_vec()).collect();
            let batch_refs: Vec<&[u8]> = batch_bytes.iter().map(|v| v.as_slice()).collect();
            black_box(futures::executor::block_on(encoder.encode_batch(&batch_refs)))
        })
    });
    
    group.finish();
}

/// Benchmark SIMD vs scalar performance
fn bench_simd_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_performance");
    
    // Vector operations benchmarks
    let vectors_a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    let vectors_b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) + 1.0).collect();
    
    group.throughput(Throughput::Elements(1024));
    group.bench_function("vector_addition_scalar", |b| {
        b.iter(|| {
            let mut result = vec![0.0f32; 1024];
            for i in 0..1024 {
                result[i] = vectors_a[i] + vectors_b[i];
            }
            black_box(result)
        })
    });
    
    group.bench_function("dot_product_scalar", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..1024 {
                sum += vectors_a[i] * vectors_b[i];
            }
            black_box(sum)
        })
    });
    
    // L2 normalization
    group.bench_function("l2_normalization", |b| {
        b.iter(|| {
            let mut data = vectors_a.clone();
            
            // Compute norm
            let norm_squared: f32 = data.iter().map(|&x| x * x).sum();
            let inv_norm = 1.0 / norm_squared.sqrt();
            
            // Normalize
            for value in &mut data {
                *value *= inv_norm;
            }
            
            black_box(data)
        })
    });
    
    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    // Sequential access
    let data: Vec<f32> = (0..100000).map(|i| i as f32).collect();
    
    group.throughput(Throughput::Elements(100000));
    group.bench_function("sequential_read", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &value in &data {
                sum += value;
            }
            black_box(sum)
        })
    });
    
    // Random access
    let indices: Vec<usize> = (0..10000).map(|i| (i * 7919) % 100000).collect();
    
    group.throughput(Throughput::Elements(10000));
    group.bench_function("random_access", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &idx in &indices {
                sum += data[idx];
            }
            black_box(sum)
        })
    });
    
    // Cache-friendly block access
    group.bench_function("block_access", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for chunk in data.chunks(64) {
                for &value in chunk {
                    sum += value;
                }
            }
            black_box(sum)
        })
    });
    
    group.finish();
}

/// Benchmark different data types and content
fn bench_content_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_types");
    
    // Natural language text
    let natural_text = "The field of natural language processing has undergone remarkable \
                       transformations in recent years, driven primarily by advances in \
                       deep learning and transformer architectures.";
    group.bench_function("natural_language", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            black_box(futures::executor::block_on(encoder.encode_text(black_box(natural_text))))
        })
    });
    // Code content
    let code_content = r#"
    fn fibonacci(n: usize) -> u64 {
        match n {
            0 => 0,
            1 => 1,
            _ => fibonacci(n - 1) + fibonacci(n - 2),
        }
    }
    "#;
    group.bench_function("source_code", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            black_box(futures::executor::block_on(encoder.encode_text(black_box(code_content))))
        })
    });
    // Structured data
    let json_content = r#"{
        "name": "Xypher",
        "version": "0.1.0",
        "features": ["cuda", "avx2", "benchmarks"],
        "performance": {
            "throughput": 1000000,
            "latency_ms": 0.1
        }
    }"#;
    group.bench_function("structured_json", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            black_box(futures::executor::block_on(encoder.encode_text(black_box(json_content))))
        })
    });
    // Multilingual content
    let multilingual_content = "Hello world! Bonjour le monde! こんにちは世界！ \
                               Hola mundo! Hallo Welt! Привет мир! 你好世界！";
    group.bench_function("multilingual", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            black_box(futures::executor::block_on(encoder.encode_text(black_box(multilingual_content))))
        })
    });
    // Binary-like content (base64)
    let binary_content = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSBiYXNlNjQgZW5jb2RlZCBzdHJpbmc=";
    group.bench_function("binary_content", |b| {
        b.iter(|| {
            let encoder = futures::executor::block_on(XypherCodex::new(128, 42)).unwrap();
            black_box(futures::executor::block_on(encoder.encode_text(black_box(binary_content))))
        })
    });
    
    group.finish();
}

/// Benchmark concurrent processing
fn bench_concurrent_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let test_texts: Vec<String> = (0..1000)
        .map(|i| format!("Test text number {} with variable content length", i))
        .collect();
    
    // Sequential processing
    group.throughput(Throughput::Elements(1000));
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let results: Vec<_> = test_texts.iter()
                .map(|text| {
                    let encoder = futures::executor::block_on(XypherCodex::new(64, 42)).unwrap();
                    futures::executor::block_on(encoder.encode_text(text.as_str()))
                })
                .collect();
            black_box(results)
        })
    });
    
    // Parallel processing with rayon
    group.bench_function("parallel_rayon", |b| {
        use rayon::prelude::*;
        
        b.iter(|| {
            let results: Vec<_> = test_texts.par_iter()
                .map(|text| {
                    let encoder = futures::executor::block_on(XypherCodex::new(64, 42)).unwrap();
                    futures::executor::block_on(encoder.encode_text(text.as_str()))
                })
                .collect();
            black_box(results)
        })
    });
    
    group.finish();
}

/// Benchmark against theoretical performance limits
fn bench_theoretical_limits(c: &mut Criterion) {
    let mut group = c.benchmark_group("theoretical_limits");
    
    // Measure theoretical E8 quantization limit
    group.bench_function("theoretical_quantization_ops", |b| {
        b.iter(|| {
            // Simulate minimal E8 operations
            let mut result = 0.0f32;
            for i in 0..8 {
                result += (i as f32).round();
            }
            black_box(result)
        })
    });
    
    // Memory bandwidth test
    group.bench_function("memory_copy_baseline", |b| {
        let src = vec![1.0f32; 1024];
        let mut dst = vec![0.0f32; 1024];
        
        b.iter(|| {
            dst.copy_from_slice(&src);
            black_box(&dst);
        })
    });
    
    // CPU instruction throughput test
    group.bench_function("cpu_arithmetic_baseline", |b| {
        b.iter(|| {
            let mut result = 1.0f32;
            for i in 0..1000 {
                result = result * 1.001 + (i as f32) * 0.001;
            }
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_e8_quantization,
    bench_tokenless_encoder,
    bench_simd_performance,
    bench_memory_patterns,
    bench_content_types,
    bench_concurrent_processing,
    bench_theoretical_limits
);

criterion_main!(benches);
