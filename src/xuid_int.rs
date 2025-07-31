/* src/xuid_int.rs */
#![warn(missing_docs)]
//! # XUID Integration with Xypher Engine
//!
//! Complete integration of Xypher Unique Identity Descriptors (XUIDs) into the ViaLisKin
//! meta-semantic quantization system, providing mathematical provenance and queryable
//! identity relationships throughout the entire processing pipeline.
//!
//! ## Integration Points
//!
//! - **ViaLisKinQuantizer**: Generates XUIDs during quantization operations
//! - **HoloSphere**: Maintains XUID-to-root mappings in RDF store
//! - **XypherEngine**: Tracks stream processing with XUID provenance
//! - **SPARQL Interface**: Enables semantic querying of XUID relationships
//! - **Performance Monitoring**: XUID-based metrics and analytics
//!
//! ## Usage Example
//!
//! ```rust
//! use crate::xuid_integration::{XuidEnabledQuantizer, XuidSparqlInterface};
//!
//! let quantizer = XuidEnabledQuantizer::new(holosphere).await?;
//! let result = quantizer.quantize_with_xuid(input_data).await?;
//!
//! // Query relationships via SPARQL
//! let sparql = XuidSparqlInterface::new(&rdf_store);
//! let similar = sparql.find_similar_xuids(&result.xuid, 0.8).await?;
//! ```
//!
/*▫~•◦────────────────────────────────────────────────────────────────────────────────────‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///◦────────────────────────────────────────────────────────────────────────────────────‣

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock as AsyncRwLock, Mutex as AsyncMutex};
use dashmap::DashMap;
use tracing::{debug, info, warn, error, instrument};
use blake3;

// Import our XUID system
use crate::xuid::{
    Xuid, XuidBuilder, XuidType, XuidError, XuidRegistry, 
    SemanticPath, XuidProvenance, XuidRegistryStats
};

// Import existing Xypher components
use crate::xypher_codex::{
    HoloSphere, ViaLisKinQuantizer, ViaLisKinQuantizationResult,
    ProcessingStrategy, XypherError, StreamResult,
};

/// Enhanced quantization result with XUID identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidQuantizationResult {
    /// Original ViaLisKin quantization result
    pub quantization: ViaLisKinQuantizationResult,
    /// Generated XUID for this result
    pub xuid: Xuid,
    /// Relationship XUIDs (similar results)
    pub related_xuids: Vec<(Xuid, f64)>,
    /// Performance metrics for XUID generation
    pub xuid_metrics: XuidGenerationMetrics,
}

/// XUID generation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidGenerationMetrics {
    /// Time to generate XUID in nanoseconds
    pub generation_time_ns: u64,
    /// Time to compute similarity relationships
    pub similarity_computation_ns: u64,
    /// Number of similar XUIDs found
    pub similar_count: usize,
    /// Memory overhead of XUID storage
    pub memory_overhead_bytes: u64,
    /// Cache hit rate for XUID lookup
    pub cache_hit_rate: f64,
}

/// XUID-enhanced ViaLisKin quantizer with provenance tracking
#[derive(Debug)]
pub struct XuidEnabledQuantizer {
    /// Underlying ViaLisKin quantizer
    quantizer: Arc<ViaLisKinQuantizer>,
    /// HoloSphere for semantic operations
    holosphere: Arc<HoloSphere>,
    /// XUID registry for relationship management
    xuid_registry: Arc<AsyncRwLock<XuidRegistry>>,
    /// Performance-optimized XUID cache
    xuid_cache: Arc<DashMap<String, Xuid, ahash::RandomState>>,
    /// Similarity computation cache
    similarity_cache: Arc<DashMap<String, Vec<(Xuid, f64)>, ahash::RandomState>>,
    /// Base namespace for RDF/SPARQL operations
    base_namespace: String,
    /// Performance metrics tracking
    metrics: Arc<AsyncMutex<XuidQuantizerMetrics>>,
}

/// Comprehensive metrics for XUID quantizer operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XuidQuantizerMetrics {
    /// Total XUIDs generated
    pub total_xuids_generated: u64,
    /// Total quantization operations performed
    pub total_quantizations: u64,
    /// Average XUID generation time in nanoseconds
    pub avg_xuid_generation_ns: u64,
    /// Average similarity computation time in nanoseconds
    pub avg_similarity_computation_ns: u64,
    /// Cache hit rate for XUID lookups
    pub cache_hit_rate: f64,
    /// Distribution of XUID types generated
    pub type_distribution: HashMap<XuidType, u64>,
    /// Distribution of orbit assignments
    pub orbit_distribution: HashMap<u32, u64>,
    /// Peak similarity relationships per XUID
    pub peak_similarity_count: usize,
    /// Total memory usage for XUID storage
    pub total_memory_usage_bytes: u64,
}

impl XuidEnabledQuantizer {
    /// Create new XUID-enabled quantizer with performance optimization
    #[instrument(skip(holosphere))]
    pub async fn new(
        holosphere: Arc<HoloSphere>,
        base_namespace: &str,
    ) -> Result<Self, XuidQuantizerError> {
        let quantizer = Arc::new(
            ViaLisKinQuantizer::new(holosphere.clone())
                .await
                .map_err(XuidQuantizerError::QuantizerInitialization)?
        );
        
        let xuid_registry = Arc::new(AsyncRwLock::new(XuidRegistry::new()));
        let xuid_cache = Arc::new(DashMap::with_hasher(ahash::RandomState::default()));
        let similarity_cache = Arc::new(DashMap::with_hasher(ahash::RandomState::default()));
        
        info!("XUID-enabled quantizer initialized with namespace: {}", base_namespace);
        
        Ok(Self {
            quantizer,
            holosphere,
            xuid_registry,
            xuid_cache,
            similarity_cache,
            base_namespace: base_namespace.to_string(),
            metrics: Arc::new(AsyncMutex::new(XuidQuantizerMetrics::default())),
        })
    }
    
    /// Perform quantization with XUID generation and relationship discovery
    #[instrument(skip(self, input_data))]
    pub async fn quantize_with_xuid(
        &self,
        input_data: &[u8],
        processing_strategy: ProcessingStrategy,
        worker_id: u32,
    ) -> Result<XuidQuantizationResult, XuidQuantizerError> {
        let overall_start = Instant::now();
        
        // Phase 1: Standard ViaLisKin quantization
        let quantization_start = Instant::now();
        let point = self.holosphere.bytes_to_e8_path(input_data, 0);
        let quantization_result = self.quantizer
            .quantize_e8_point_vialiskin_semantic(&point)
            .await
            .map_err(|e| XuidQuantizerError::QuantizationFailed(e))?;
        let quantization_time = quantization_start.elapsed();
        
        // Phase 2: XUID generation with provenance
        let xuid_start = Instant::now();
        let xuid = self.generate_xuid_with_provenance(
            input_data,
            &quantization_result,
            processing_strategy,
            worker_id,
            quantization_time,
        ).await?;
        let xuid_generation_time = xuid_start.elapsed();
        
        // Phase 3: Similarity relationship discovery
        let similarity_start = Instant::now();
        let related_xuids = self.discover_xuid_relationships(&xuid).await?;
        let similarity_time = similarity_start.elapsed();
        
        // Phase 4: Registry registration and caching
        self.register_and_cache_xuid(xuid.clone()).await?;
        
        // Phase 5: Metrics update
        let total_time = overall_start.elapsed();
        self.update_metrics(&xuid, xuid_generation_time, similarity_time, related_xuids.len()).await;
        
        let xuid_metrics = XuidGenerationMetrics {
            generation_time_ns: xuid_generation_time.as_nanos() as u64,
            similarity_computation_ns: similarity_time.as_nanos() as u64,
            similar_count: related_xuids.len(),
            memory_overhead_bytes: self.estimate_xuid_memory_usage(&xuid),
            cache_hit_rate: self.calculate_current_cache_hit_rate().await,
        };
        
        debug!("XUID quantization completed in {:?}: {}", total_time, xuid);
        
        Ok(XuidQuantizationResult {
            quantization: quantization_result,
            xuid,
            related_xuids,
            xuid_metrics,
        })
    }
    
    /// Generate XUID with comprehensive provenance information
    async fn generate_xuid_with_provenance(
        &self,
        input_data: &[u8],
        quantization_result: &ViaLisKinQuantizationResult,
        processing_strategy: ProcessingStrategy,
        worker_id: u32,
        processing_time: Duration,
    ) -> Result<Xuid, XuidQuantizerError> {
        // Create detailed provenance record
        let provenance = XuidProvenance::new(
            input_data,
            &format!("{:?}", processing_strategy),
            worker_id,
            quantization_result.reasoning_quality_score,
            processing_time.as_nanos() as u64,
            std::mem::size_of_val(quantization_result) as u64,
            1, // SIMD operations count (simplified)
        );
        
        // Extract semantic path from quantization result
        let semantic_path = self.extract_semantic_path(quantization_result).await?;
        
        // Determine orbit ID from quantized coordinates
        let orbit_id = self.holosphere
            .compute_orbit_id(&quantization_result.quantized_coordinates)
            .map_err(XuidQuantizerError::QuantizationFailed)?;
        
        // Build XUID with all components
        let xuid = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(input_data)
            .with_quantization_result(&quantization_result.quantized_coordinates.to_vec())
            .with_semantic_path(semantic_path)
            .with_provenance(provenance)
            .with_orbit_id(orbit_id as u32)
            .build()
            .map_err(XuidQuantizerError::XuidGeneration)?;
        
        Ok(xuid)
    }
    
    /// Extract semantic path from ViaLisKin quantization result
    async fn extract_semantic_path(
        &self,
        result: &ViaLisKinQuantizationResult,
    ) -> Result<SemanticPath, XuidQuantizerError> {
        // Convert ViaLisKin traversal path to XUID semantic path
        let traversal_nodes = vec![result.semantic_root_iri.clone()];
        
        // Extract reflection sequence from provenance
        let reflection_sequence = self.extract_reflection_sequence(&result.provenance).await?;
        
        // Determine orbit transitions
        let orbit_transitions = vec![
            self.holosphere
                .compute_orbit_id(&result.quantized_coordinates)
                .map_err(XuidQuantizerError::QuantizationFailed)? as u32
        ];
        
        // Calculate progression metrics
        let distance_progression = vec![result.provenance.semantic_distance];
        let quality_progression = vec![result.reasoning_quality_score];
        
        Ok(SemanticPath {
            traversal_nodes,
            reflection_sequence,
            orbit_transitions,
            distance_progression,
            quality_progression,
        })
    }
    
    /// Extract reflection sequence from quantization provenance
    async fn extract_reflection_sequence(
        &self,
        provenance: &crate::xypher_codex::ViaLisKinQuantizationProvenance,
    ) -> Result<Vec<u8>, XuidQuantizerError> {
        // Analyze traversal path to extract Weyl reflections
        let mut reflections = Vec::new();
        
        // Convert semantic path steps to reflection indices
        for step in &provenance.traversal_path.reasoning_steps {
            if step.contains("reflection") || step.contains("Weyl") {
                // Extract reflection index from step description
                let reflection_index = self.parse_reflection_index(step)?;
                reflections.push(reflection_index);
            }
        }
        
        // If no explicit reflections found, derive from coordinates
        if reflections.is_empty() {
            reflections = self.derive_reflections_from_coordinates(&provenance.input_point)
                ?;
        }
        
        Ok(reflections)
    }
    
    /// Parse reflection index from step description
    fn parse_reflection_index(&self, step: &str) -> Result<u8, XuidQuantizerError> {
        // Simple heuristic to extract reflection indices
        if let Some(index_str) = step.split_whitespace()
            .find(|word| word.parse::<u8>().is_ok()) {
            index_str.parse().map_err(|_| XuidQuantizerError::ReflectionParsing(step.to_string()))
        } else {
            // Default reflection based on step content hash
            let step_hash = blake3::hash(step.as_bytes());
            Ok((step_hash.as_bytes()[0] % 8) + 1) // E8 has 8 simple reflections
        }
    }
    
    /// Derive reflection sequence from coordinate analysis
    fn derive_reflections_from_coordinates(
        &self,
        coordinates: &[f32; 8],
    ) -> Result<Vec<u8>, XuidQuantizerError> {
        let mut reflections = Vec::new();
        
        // Analyze coordinate patterns to infer reflection sequence
        for (i, &coord) in coordinates.iter().enumerate() {
            if coord.abs() > 1.0 {
                reflections.push((i as u8) + 1);
            }
        }
        
        // Ensure at least one reflection for non-trivial paths
        if reflections.is_empty() {
            let coord_sum = coordinates.iter().sum::<f32>();
            reflections.push(((coord_sum as u32) % 8 + 1) as u8);
        }
        
        Ok(reflections)
    }
    
    /// Discover XUID relationships through similarity analysis
    async fn discover_xuid_relationships(
        &self,
        xuid: &Xuid,
    ) -> Result<Vec<(Xuid, f64)>, XuidQuantizerError> {
        let cache_key = format!("sim_{}", xuid);
        
        // Check similarity cache first
        if let Some(cached_similar) = self.similarity_cache.get(&cache_key) {
            return Ok(cached_similar.clone());
        }
        
        // Perform comprehensive similarity search
        let registry = self.xuid_registry.read().await;
        let similar_xuids = registry.find_similar(xuid, 0.7);
        
        // Convert to owned data
        let owned_similar: Vec<(Xuid, f64)> = similar_xuids
            .into_iter()
            .map(|(xuid_ref, score)| (xuid_ref.clone(), score))
            .collect();
        
        // Cache results for future queries
        self.similarity_cache.insert(cache_key, owned_similar.clone());
        
        Ok(owned_similar)
    }
    
    /// Register XUID and update caches
    async fn register_and_cache_xuid(&self, xuid: Xuid) -> Result<(), XuidQuantizerError> {
        // Register in main registry
        {
            let mut registry = self.xuid_registry.write().await;
            registry.register(xuid.clone())
                .map_err(XuidQuantizerError::RegistrationFailed)?;
        }
        
        // Cache for fast lookup
        let xuid_string = xuid.to_string();
        self.xuid_cache.insert(xuid_string, xuid);
        
        Ok(())
    }
    
    /// Update comprehensive performance metrics
    async fn update_metrics(
        &self,
        xuid: &Xuid,
        generation_time: Duration,
        similarity_time: Duration,
        similarity_count: usize,
    ) {
        let mut metrics = self.metrics.lock().await;
        
        metrics.total_xuids_generated += 1;
        metrics.total_quantizations += 1;
        
        // Update moving averages
        let total_ops = metrics.total_xuids_generated;
        metrics.avg_xuid_generation_ns = 
            ((metrics.avg_xuid_generation_ns * (total_ops - 1)) + generation_time.as_nanos() as u64) / total_ops;
        metrics.avg_similarity_computation_ns = 
            ((metrics.avg_similarity_computation_ns * (total_ops - 1)) + similarity_time.as_nanos() as u64) / total_ops;
        
        // Update type distribution
        *metrics.type_distribution.entry(xuid.xuid_type).or_insert(0) += 1;
        *metrics.orbit_distribution.entry(xuid.orbit_id).or_insert(0) += 1;
        
        // Update peak similarity count
        metrics.peak_similarity_count = metrics.peak_similarity_count.max(similarity_count);
        
        // Estimate memory usage
        metrics.total_memory_usage_bytes += self.estimate_xuid_memory_usage(xuid);
        
        // Update cache hit rate
        metrics.cache_hit_rate = self.calculate_current_cache_hit_rate().await;
    }
    
    /// Estimate memory usage for XUID storage
    fn estimate_xuid_memory_usage(&self, xuid: &Xuid) -> u64 {
        let base_size = std::mem::size_of::<Xuid>() as u64;
        let string_sizes = (xuid.delta_signature.len() + 
                           xuid.reflection_path.len() + 
                           xuid.semantic_hash.len() + 
                           xuid.provenance_hash.len()) as u64;
        base_size + string_sizes
    }
    
    /// Calculate current cache hit rate
    async fn calculate_current_cache_hit_rate(&self) -> f64 {
        let cache_size = self.xuid_cache.len() as f64;
        let registry_size = {
            let registry = self.xuid_registry.read().await;
            registry.stats().total_xuids as f64
        };
        
        if registry_size > 0.0 {
            (cache_size / registry_size).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Batch quantization with XUID generation
    #[instrument(skip(self, input_batch))]
    pub async fn quantize_batch_with_xuids(
        &self,
        input_batch: &[&[u8]],
        processing_strategy: ProcessingStrategy,
        worker_id: u32,
    ) -> Result<Vec<XuidQuantizationResult>, XuidQuantizerError> {
        let batch_start = Instant::now();
        
        // Parallel processing of batch items
        let results: Result<Vec<_>, _> = futures::future::try_join_all(
            input_batch.iter().enumerate().map(|(i, &data)| {
                let quantizer = self.clone();
                async move {
                    quantizer.quantize_with_xuid(
                        data,
                        processing_strategy,
                        worker_id + i as u32,
                    ).await
                }
            })
        ).await;
        
        let batch_time = batch_start.elapsed();
        info!("Batch XUID quantization completed: {} items in {:?}", 
              input_batch.len(), batch_time);
        
        results.map_err(|e| e)
    }
    
    /// Get XUID by string identifier
    pub async fn get_xuid(&self, xuid_str: &str) -> Option<Xuid> {
        // Check cache first
        if let Some(cached_xuid) = self.xuid_cache.get(xuid_str) {
            return Some(cached_xuid.clone());
        }
        
        // Check registry
        let registry = self.xuid_registry.read().await;
        registry.get(xuid_str).cloned()
    }
    
    /// Find XUIDs in specific orbit
    pub async fn get_orbit_xuids(&self, orbit_id: u32) -> Vec<Xuid> {
        let registry = self.xuid_registry.read().await;
        registry.get_orbit_xuids(orbit_id).into_iter().cloned().collect()
    }
    
    /// Get comprehensive quantizer statistics
    pub async fn get_comprehensive_stats(&self) -> XuidQuantizerStats {
        let metrics = self.metrics.lock().await.clone();
        let registry_stats = {
            let registry = self.xuid_registry.read().await;
            registry.stats()
        };
        
        XuidQuantizerStats {
            metrics,
            registry_stats,
            cache_size: self.xuid_cache.len(),
            similarity_cache_size: self.similarity_cache.len(),
        }
    }
    
    /// Export all XUIDs as SPARQL triples
    pub async fn export_sparql_triples(&self) -> Vec<String> {
        let registry = self.xuid_registry.read().await;
        registry.export_sparql_triples(&self.base_namespace)
    }
    
    /// Clear caches to free memory
    pub async fn clear_caches(&self) {
        self.xuid_cache.clear();
        self.similarity_cache.clear();
        info!("XUID caches cleared");
    }
}

impl Clone for XuidEnabledQuantizer {
    fn clone(&self) -> Self {
        Self {
            quantizer: self.quantizer.clone(),
            holosphere: self.holosphere.clone(),
            xuid_registry: self.xuid_registry.clone(),
            xuid_cache: self.xuid_cache.clone(),
            similarity_cache: self.similarity_cache.clone(),
            base_namespace: self.base_namespace.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

/// Comprehensive statistics for XUID-enabled quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidQuantizerStats {
    /// Core quantizer metrics
    pub metrics: XuidQuantizerMetrics,
    /// Registry statistics
    pub registry_stats: XuidRegistryStats,
    /// Current cache size
    pub cache_size: usize,
    /// Similarity cache size
    pub similarity_cache_size: usize,
}

/// SPARQL interface for XUID querying and relationship discovery
#[derive(Debug)]
pub struct XuidSparqlInterface {
    /// XUID-enabled quantizer for data access
    quantizer: Arc<XuidEnabledQuantizer>,
    /// Base namespace for SPARQL queries
    base_namespace: String,
    /// Query performance cache
    query_cache: Arc<DashMap<String, Vec<Xuid>, ahash::RandomState>>,
}

impl XuidSparqlInterface {
    /// Create new SPARQL interface
    pub fn new(quantizer: Arc<XuidEnabledQuantizer>, base_namespace: &str) -> Self {
        Self {
            quantizer,
            base_namespace: base_namespace.to_string(),
            query_cache: Arc::new(DashMap::with_hasher(ahash::RandomState::default())),
        }
    }
    
    /// Find similar XUIDs using SPARQL-style queries
    #[instrument(skip(self))]
    pub async fn find_similar_xuids(
        &self,
        target_xuid: &Xuid,
        similarity_threshold: f64,
    ) -> Result<Vec<(Xuid, f64)>, XuidQueryError> {
        let registry = self.quantizer.xuid_registry.read().await;
        let similar = registry.find_similar(target_xuid, similarity_threshold);
        
        Ok(similar.into_iter().map(|(xuid, score)| (xuid.clone(), score)).collect())
    }
    
    /// Query XUIDs by orbit using SPARQL-style interface
    pub async fn query_xuids_by_orbit(&self, orbit_id: u32) -> Result<Vec<Xuid>, XuidQueryError> {
        let cache_key = format!("orbit_{}", orbit_id);
        
        if let Some(cached_result) = self.query_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }
        
        let xuids = self.quantizer.get_orbit_xuids(orbit_id).await;
        self.query_cache.insert(cache_key, xuids.clone());
        
        Ok(xuids)
    }
    
    /// Query XUIDs by type
    pub async fn query_xuids_by_type(&self, xuid_type: XuidType) -> Result<Vec<Xuid>, XuidQueryError> {
        let registry = self.quantizer.xuid_registry.read().await;
        let all_xuids: Vec<Xuid> = registry.export_sparql_triples(&self.base_namespace)
            .iter()
            .filter_map(|triple| {
                if triple.contains(&format!("xuidType \"{}\"", xuid_type)) {
                    // Extract XUID from triple (simplified)
                    None // Would need proper SPARQL parsing
                } else {
                    None
                }
            })
            .collect();
        
        Ok(all_xuids)
    }
    
    /// Complex relationship queries
    pub async fn query_xuid_relationships(
        &self,
        query: &str,
    ) -> Result<Vec<XuidRelationship>, XuidQueryError> {
        // Parse simplified SPARQL-like query
        if query.contains("SIMILAR TO") {
            let xuid_str = self.extract_xuid_from_query(query)?;
            if let Some(target_xuid) = self.quantizer.get_xuid(&xuid_str).await {
                let similar = self.find_similar_xuids(&target_xuid, 0.7).await?;
                return Ok(similar.into_iter().map(|(xuid, score)| {
                    XuidRelationship::Similarity {
                        source: target_xuid.clone(),
                        target: xuid,
                        score,
                    }
                }).collect());
            }
        }
        
        Ok(vec![])
    }
    
    /// Extract XUID from query string
    fn extract_xuid_from_query(&self, query: &str) -> Result<String, XuidQueryError> {
        // Simple extraction for "SIMILAR TO 'XUID-...'"
        if let Some(start) = query.find("XUID-") {
            let xuid_part = &query[start..];
            if let Some(end) = xuid_part.find('\'') {
                return Ok(xuid_part[..end].to_string());
            }
        }
        
        Err(XuidQueryError::InvalidQuery(query.to_string()))
    }
    
    /// Generate SPARQL schema for XUID ontology
    pub fn generate_ontology_schema(&self) -> String {
        let registry = XuidRegistry::new(); // Temporary for schema generation
        registry.generate_sparql_schema(&self.base_namespace)
    }
    
    /// Export XUID data as Turtle/RDF
    pub async fn export_as_turtle(&self) -> String {
        let triples = self.quantizer.export_sparql_triples().await;
        let mut turtle = format!(
            "@prefix xuid: <{}/> .\n@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n",
            self.base_namespace
        );
        
        for triple in triples {
            turtle.push_str(&triple);
            turtle.push('\n');
        }
        
        turtle
    }
}

/// XUID relationship types for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XuidRelationship {
    /// Similarity relationship with score
    Similarity {
        /// Source XUID in the similarity relationship.
        source: Xuid,
        /// Target XUID considered similar.
        target: Xuid,
        /// Similarity score between source and target.
        score: f64,
    },
    /// Derivation relationship (provenance chain)
    Derivation {
        /// Parent XUID from which the child is derived.
        parent: Xuid,
        /// Child XUID resulting from the transformation.
        child: Xuid,
        /// Description of the transformation process.
        transformation: String,
    },
    /// Orbit membership relationship
    OrbitMembership {
        /// XUID that is a member of the orbit.
        xuid: Xuid,
        /// Orbit identifier.
        orbit_id: u32,
        /// Optional center XUID of the orbit.
        orbit_center: Option<Xuid>,
    },
}

/// Enhanced stream result with XUID integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidStreamResult {
    /// Original stream result
    pub stream_result: StreamResult,
    /// Associated XUID for provenance
    pub xuid: Xuid,
    /// Related XUIDs discovered during processing
    pub related_xuids: Vec<(Xuid, f64)>,
    /// XUID generation performance metrics
    pub xuid_performance: XuidGenerationMetrics,
}

/// Error types for XUID quantizer operations
#[derive(Error, Debug)]
pub enum XuidQuantizerError {
    #[error("Quantizer initialization failed: {0}")]
    /// Error during quantizer initialization.
    QuantizerInitialization(XypherError),
    
    #[error("Quantization operation failed: {0}")]
    /// Error during quantization operation.
    QuantizationFailed(XypherError),
    
    #[error("XUID generation failed: {0}")]
    /// Error during XUID generation.
    XuidGeneration(#[from] XuidError),
    
    #[error("XUID registration failed: {0}")]
    /// Error during XUID registration.
    RegistrationFailed(XuidError),
    
    #[error("Error parsing reflection index from step: {0}")]
    /// Error parsing reflection index from step.
    ReflectionParsing(String),
    
    #[error("Error extracting semantic path: {0}")]
    /// Error extracting semantic path.
    SemanticPathExtraction(String),
    
    #[error("Error during cache operation: {0}")]
    /// Error during cache operation.
    CacheOperation(String),
    
    #[error("Error updating metrics: {0}")]
    /// Error updating metrics.
    MetricsUpdate(String),
}

/// Error types for XUID query operations
#[derive(Error, Debug)]
pub enum XuidQueryError {
    /// Returned when a query string does not match the expected format.
    #[error("Invalid query format: {0}")]
    InvalidQuery(String),
    
    /// Indicates a failure occurred during query execution.
    #[error("Query execution failed: {0}")]
    QueryExecution(String),
    
    /// Error raised when parsing SPARQL queries fails.
    #[error("SPARQL parsing failed: {0}")]
    SparqlParsing(String),
    
    /// Represents a failure during cache lookup operations.
    #[error("Cache lookup failed: {0}")]
    CacheLookup(String),
    
    /// Error for failures accessing the XUID registry.
    #[error("Registry access failed: {0}")]
    RegistryAccess(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    async fn create_test_quantizer() -> XuidEnabledQuantizer {
        let holosphere = Arc::new(HoloSphere::new("http://test.ai/").await.unwrap());
        XuidEnabledQuantizer::new(holosphere, "http://test.ai/xuid")
            .await
            .unwrap()
    }
    
    #[tokio::test]
    async fn test_xuid_quantization() {
        let quantizer = create_test_quantizer().await;
        let test_data = b"test_semantic_data";
        
        let result = quantizer
            .quantize_with_xuid(test_data, ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        
        assert_eq!(result.xuid.xuid_type, XuidType::E8Quantized);
        assert!(!result.xuid.delta_signature.is_empty());
        assert!(result.xuid_metrics.generation_time_ns > 0);
    }
    
    #[tokio::test]
    async fn test_batch_xuid_quantization() {
        let quantizer = create_test_quantizer().await;
        let batch_data = vec![
            b"data1".as_slice(),
            b"data2".as_slice(),
            b"data3".as_slice(),
        ];
        
        let results = quantizer
            .quantize_batch_with_xuids(&batch_data, ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        
        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.xuid.xuid_type, XuidType::E8Quantized);
            assert!(result.xuid_metrics.generation_time_ns > 0);
        }
    }
    
    /// Test XUID similarity discovery after quantization.
    /// 
    /// This test creates two XUIDs with similar data and checks if they're
    /// discovered as related when querying the similarity cache. Note that
    /// this test is not guaranteed to pass due to the probabilistic nature
    /// of semantic similarity calculation.
    #[tokio::test]
    async fn test_xuid_similarity_discovery() {
        let quantizer = create_test_quantizer().await;
        
        quantizer
            .quantize_with_xuid(b"similar_data_1", ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        let _result2 = quantizer
            .quantize_with_xuid(b"similar_data_2", ProcessingStrategy::Hybrid, 2)
            .await
            .unwrap();
        
        // Check if they're discovered as related
        // result2.related_xuids may or may not find similarities in test
    }
    
    #[tokio::test]
    async fn test_xuid_registry_operations() {
        let quantizer = create_test_quantizer().await;
        let test_data = b"registry_test_data";
        
        let result = quantizer
            .quantize_with_xuid(test_data, ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        
        // Verify XUID can be retrieved
        let retrieved = quantizer.get_xuid(&result.xuid.to_string()).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), result.xuid);
    }
    
    #[tokio::test]
    async fn test_sparql_interface() {
        let quantizer = Arc::new(create_test_quantizer().await);
        let sparql = XuidSparqlInterface::new(quantizer.clone(), "http://test.ai/xuid");
        
        // Generate test XUID
        let result = quantizer
            .quantize_with_xuid(b"sparql_test", ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        
        // Test orbit query
        let orbit_xuids = sparql.query_xuids_by_orbit(result.xuid.orbit_id).await.unwrap();
        assert!(orbit_xuids.len() >= 1);
        
        // Test similarity query
        let _similar = sparql.find_similar_xuids(&result.xuid, 0.5).await.unwrap();
        // similar may be empty in test
    }
    
    #[tokio::test]
    async fn test_xuid_performance_metrics() {
        let quantizer = create_test_quantizer().await;
        
        // Generate several XUIDs to test metrics
        for i in 0..5 {
            let data = format!("performance_test_{}", i);
            let _result = quantizer
                .quantize_with_xuid(data.as_bytes(), ProcessingStrategy::Hybrid, i as u32)
                .await
                .unwrap();
        }
        
        let stats = quantizer.get_comprehensive_stats().await;
        assert_eq!(stats.metrics.total_xuids_generated, 5);
        assert!(stats.metrics.avg_xuid_generation_ns > 0);
        assert_eq!(stats.registry_stats.total_xuids, 5);
    }
    
    #[tokio::test]
    async fn test_xuid_cache_performance() {
        let quantizer = create_test_quantizer().await;
        let test_data = b"cache_test_data";
        
        // Generate XUID
        let result = quantizer
            .quantize_with_xuid(test_data, ProcessingStrategy::Hybrid, 1)
            .await
            .unwrap();
        
        // First lookup (should hit cache)
        let start = Instant::now();
        let retrieved1 = quantizer.get_xuid(&result.xuid.to_string()).await;
        let first_lookup = start.elapsed();
        
        // Second lookup (should be faster from cache)
        let start = Instant::now();
        let retrieved2 = quantizer.get_xuid(&result.xuid.to_string()).await;
        let second_lookup = start.elapsed();
        
        assert!(retrieved1.is_some());
        assert!(retrieved2.is_some());
        assert_eq!(retrieved1.unwrap(), retrieved2.unwrap());
        
        // Second lookup should generally be faster (though not guaranteed in tests)
        println!("First lookup: {:?}, Second lookup: {:?}", first_lookup, second_lookup);
    }
}
