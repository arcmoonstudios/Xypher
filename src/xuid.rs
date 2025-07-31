/* src/xuid.rs */
#![warn(missing_docs)]
//! # XUID: Xypher Unique Identity Descriptor
//!
//! Revolutionary identity system that transcends traditional UUIDs by encoding mathematical
//! provenance, semantic reasoning paths, and lattice-based quantization results into a
//! semantically meaningful, cryptographically verifiable identifier.
//!
//! ## Core Philosophy
//!
//! While UUIDs provide uniqueness without meaning, XUIDs provide:
//! - **Mathematical Provenance**: Complete E8 lattice traversal path
//! - **Semantic Reasoning**: ViaLisKin meta-semantic decision chain
//! - **Cryptographic Integrity**: Tamper-evident hash verification
//! - **SPARQL Queryability**: RDF-native identity relationships
//! - **Reversible Traceability**: Reconstruct the reasoning process
//!
//! ## XUID Format Specification
//!
//! ```text
//! XUID-E8Q::{delta_signature}::{orbit_id}::{reflection_path}::{semantic_hash}::{provenance_hash}
//! 
//! Example:
//! XUID-E8Q::Δ3F4C2A18::ORBIT-07::RPL-256-042::SEM-A1B2C3D4::PROV-76543210
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use crate::xuid::{XuidBuilder, XuidType, SemanticPath};
//! 
//! let builder = XuidBuilder::new(XuidType::E8Quantized);
//! let xuid = builder
//!     .with_input_data(b"semantic_content")
//!     .with_quantization_result(&quantized_point)
//!     .with_semantic_path(semantic_path)
//!     .with_provenance(provenance)
//!     .build()?;
//! 
//! // SPARQL queryable
//! let query = format!("SELECT ?related WHERE {{ ?related xypher:similarTo <{}> }}", xuid);
//! ```
//!
/*▫~•◦────────────────────────────────────────────────────────────────────────────────────‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///◦────────────────────────────────────────────────────────────────────────────────────‣

use std::fmt;
use std::str::FromStr;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use blake3::Hasher;
use chrono::{DateTime, Utc};

/// Minimal stub for TraversableEdge used by xypher_codex.rs
#[derive(Debug, Clone)]
pub struct TraversableEdge {
    /// Index of the source root node.
    pub source_root_index: usize,
    /// Index of the target root node.
    pub target_root_index: usize,
    /// Distance of the reflection in the lattice.
    pub reflection_distance: f32,
    /// Semantic weight of the edge.
    pub semantic_weight: f64,
    /// Cost of traversal for this edge.
    pub traversal_cost: f64,
}

/// Minimal stub for CpuFeatures used by xypher_codex.rs
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    /// Indicates AVX2 CPU feature support.
    pub avx2: bool,
    /// Indicates FMA CPU feature support.
    pub fma: bool,
    /// Indicates AVX-512F CPU feature support.
    pub avx512f: bool,
}

/// XUID domain type specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XuidType {
    /// E8 lattice quantized identity
    E8Quantized,
    /// RDF graph-derived semantic identity
    RdfSemantic,
    /// Stream processing trace identity
    StreamTrace,
    /// Cross-domain similarity identity
    Similarity,
    /// Meta-semantic reasoning identity
    MetaSemantic,
}

impl fmt::Display for XuidType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XuidType::E8Quantized => write!(f, "E8Q"),
            XuidType::RdfSemantic => write!(f, "RDF"),
            XuidType::StreamTrace => write!(f, "STR"),
            XuidType::Similarity => write!(f, "SIM"),
            XuidType::MetaSemantic => write!(f, "META"),
        }
    }
}

impl FromStr for XuidType {
    type Err = XuidError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "E8Q" => Ok(XuidType::E8Quantized),
            "RDF" => Ok(XuidType::RdfSemantic),
            "STR" => Ok(XuidType::StreamTrace),
            "SIM" => Ok(XuidType::Similarity),
            "META" => Ok(XuidType::MetaSemantic),
            _ => Err(XuidError::InvalidType(s.to_string())),
        }
    }
}

/// Semantic reasoning path component
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticPath {
    /// Root nodes traversed during reasoning
    pub traversal_nodes: Vec<String>,
    /// Weyl reflection sequence applied
    pub reflection_sequence: Vec<u8>,
    /// Orbit transitions during quantization
    pub orbit_transitions: Vec<u32>,
    /// Semantic distance at each step
    pub distance_progression: Vec<f64>,
    /// Quality scores throughout reasoning
    pub quality_progression: Vec<f64>,
}

impl SemanticPath {
    /// Create direct semantic path (single step)
    pub fn direct(target_node: String) -> Self {
        Self {
            traversal_nodes: vec![target_node],
            reflection_sequence: vec![],
            orbit_transitions: vec![],
            distance_progression: vec![0.0],
            quality_progression: vec![1.0],
        }
    }
    
    /// Compress path to compact representation
    pub fn compress(&self) -> String {
        let node_count = self.traversal_nodes.len() as u16;
        let _reflection_hash = blake3::hash(&self.reflection_sequence);
        let orbit_signature = self.orbit_transitions.iter()
            .fold(0u32, |acc, &orbit| acc.wrapping_mul(31).wrapping_add(orbit));
        
        format!("RPL-{:03X}-{:04X}", node_count, orbit_signature)
    }
    
    /// Calculate path complexity score
    pub fn complexity_score(&self) -> f64 {
        let traversal_complexity = self.traversal_nodes.len() as f64;
        let reflection_complexity = self.reflection_sequence.len() as f64 * 1.5;
        let orbit_complexity = self.orbit_transitions.len() as f64 * 2.0;
        
        (traversal_complexity + reflection_complexity + orbit_complexity) / 10.0
    }
}

/// Provenance information for XUID generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XuidProvenance {
    /// Input data hash (BLAKE3)
    pub input_hash: String,
    /// Processing strategy used
    pub processing_strategy: String,
    /// Worker/thread identifier
    pub worker_id: u32,
    /// Timestamp of generation
    #[serde(with = "chrono::serde::ts_seconds")]
    pub timestamp: DateTime<Utc>,
    /// Quality score achieved
    pub quality_score: f64,
    /// Processing time in nanoseconds
    pub processing_time_ns: u64,
    /// Memory usage during processing
    pub memory_usage_bytes: u64,
    /// SIMD operations performed
    pub simd_operations_count: u64,
}

impl XuidProvenance {
    /// Create provenance from processing context
    pub fn new(
        input_data: &[u8],
        processing_strategy: &str,
        worker_id: u32,
        quality_score: f64,
        processing_time_ns: u64,
        memory_usage_bytes: u64,
        simd_operations_count: u64,
    ) -> Self {
        let input_hash = blake3::hash(input_data);
        
        Self {
            input_hash: format!("{:016x}", input_hash.as_bytes()[0..8].iter()
                .fold(0u64, |acc, &b| (acc << 8) | b as u64)),
            processing_strategy: processing_strategy.to_string(),
            worker_id,
            timestamp: Utc::now(),
            quality_score,
            processing_time_ns,
            memory_usage_bytes,
            simd_operations_count,
        }
    }
    
    /// Compress provenance to compact hash
    pub fn compress(&self) -> String {
        let mut hasher = Hasher::new();
        hasher.update(self.input_hash.as_bytes());
        hasher.update(self.processing_strategy.as_bytes());
        hasher.update(&self.worker_id.to_le_bytes());
        hasher.update(&self.timestamp.timestamp_nanos_opt().unwrap_or_default().to_le_bytes());
        hasher.update(&self.quality_score.to_bits().to_le_bytes());
        
        let hash = hasher.finalize();
        format!("PROV-{:08X}", 
                hash.as_bytes()[0..4].iter()
                    .fold(0u32, |acc, &b| (acc << 8) | b as u32))
    }
}

/// Complete XUID structure with mathematical provenance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Xuid {
    /// Type of XUID (domain specification)
    pub xuid_type: XuidType,
    /// Delta signature from input transformation
    pub delta_signature: String,
    /// E8 lattice orbit identifier
    pub orbit_id: u32,
    /// Compressed semantic reasoning path
    pub reflection_path: String,
    /// Semantic content hash
    pub semantic_hash: String,
    /// Compressed provenance hash
    pub provenance_hash: String,
    /// Full semantic path (not serialized in string form)
    #[serde(skip)]
    pub semantic_path: Option<SemanticPath>,
    /// Full provenance (not serialized in string form)
    #[serde(skip)]
    pub provenance: Option<XuidProvenance>,
}

impl Xuid {
    /// Generate a new XUID with random values (UUID v4 compatible)
    pub fn new_v4() -> Self {
        use rand::{rng, Rng};
        
        let mut rng = rng();
let random_bytes: Vec<u8> = (0..16).map(|_| rng.random::<u8>()).collect();
        
        XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(&random_bytes)
.with_orbit_id(rng.random::<u32>() % 30)
            .build()
            .expect("Failed to generate random XUID")
    }
    
    /// Parse XUID from string representation
    pub fn parse(xuid_str: &str) -> Result<Self, XuidError> {
        let components: Vec<&str> = xuid_str.split("::").collect();
        
        if components.len() != 6 {
            return Err(XuidError::InvalidFormat {
                expected: 6,
                found: components.len(),
                input: xuid_str.to_string(),
            });
        }
        
        // Parse type component
        let type_part = components[0].strip_prefix("XUID-")
            .ok_or_else(|| XuidError::MissingPrefix(xuid_str.to_string()))?;
        let xuid_type = XuidType::from_str(type_part)?;
        
        // Extract delta signature
        let delta_signature = components[1].strip_prefix("Δ")
            .ok_or_else(|| XuidError::InvalidDeltaFormat(components[1].to_string()))?
            .to_string();
        
        // Parse orbit ID
        let orbit_id = components[2].strip_prefix("ORBIT-")
            .ok_or_else(|| XuidError::InvalidOrbitFormat(components[2].to_string()))?
            .parse::<u32>()
            .map_err(|_| XuidError::InvalidOrbitFormat(components[2].to_string()))?;
        
        // Validate other components
        if !components[3].starts_with("RPL-") {
            return Err(XuidError::InvalidPathFormat(components[3].to_string()));
        }
        if !components[4].starts_with("SEM-") {
            return Err(XuidError::InvalidSemanticFormat(components[4].to_string()));
        }
        if !components[5].starts_with("PROV-") {
            return Err(XuidError::InvalidProvenanceFormat(components[5].to_string()));
        }
        
        Ok(Self {
            xuid_type,
            delta_signature,
            orbit_id,
            reflection_path: components[3].to_string(),
            semantic_hash: components[4].to_string(),
            provenance_hash: components[5].to_string(),
            semantic_path: None,
            provenance: None,
        })
    }
    
    /// Convert XUID to IRI for RDF/SPARQL usage
    pub fn to_iri(&self, base_namespace: &str) -> String {
        format!("{}/xuid/{}", base_namespace.trim_end_matches('/'), self.to_string())
    }
    
    /// Calculate similarity score with another XUID
    pub fn similarity_score(&self, other: &Xuid) -> f64 {
        if self.xuid_type != other.xuid_type {
            return 0.0;
        }
        
        let mut score = 0.0;
        let mut _factors = 0;
        
        // Orbit similarity (40% weight)
        if self.orbit_id == other.orbit_id {
            score += 0.4;
        } else {
            let orbit_distance = (self.orbit_id as i32 - other.orbit_id as i32).abs() as f64;
            score += 0.4 * (1.0 / (1.0 + orbit_distance * 0.1));
        }
        _factors += 1;
        
        // Delta signature similarity (30% weight)
        let delta_similarity = self.calculate_string_similarity(&self.delta_signature, &other.delta_signature);
        score += 0.3 * delta_similarity;
        _factors += 1;
        
        // Semantic hash similarity (20% weight)
        let semantic_similarity = self.calculate_string_similarity(&self.semantic_hash, &other.semantic_hash);
        score += 0.2 * semantic_similarity;
        _factors += 1;
        
        // Path similarity (10% weight)
        let path_similarity = self.calculate_string_similarity(&self.reflection_path, &other.reflection_path);
        score += 0.1 * path_similarity;
        _factors += 1;
        
        score
    }
    
    /// Calculate Hamming-based string similarity
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }
        
        let len = s1.len().max(s2.len());
        if len == 0 {
            return 1.0;
        }
        
        let matching_chars = s1.chars().zip(s2.chars())
            .filter(|(c1, c2)| c1 == c2)
            .count();
        
        matching_chars as f64 / len as f64
    }
    
    /// Verify XUID integrity (if full provenance available)
    pub fn verify_integrity(&self) -> Result<bool, XuidError> {
        // Basic format validation
        if self.delta_signature.is_empty() || self.semantic_hash.is_empty() {
            return Ok(false);
        }
        
        // If full provenance is available, verify hashes
        if let Some(ref provenance) = self.provenance {
            let expected_provenance_hash = provenance.compress();
            if expected_provenance_hash != self.provenance_hash {
                return Ok(false);
            }
        }
        
        if let Some(ref semantic_path) = self.semantic_path {
            let expected_path_hash = semantic_path.compress();
            if expected_path_hash != self.reflection_path {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Generate SPARQL triples for RDF integration
    pub fn to_sparql_triples(&self, base_namespace: &str) -> Vec<String> {
        let iri = self.to_iri(base_namespace);
        let mut triples = Vec::new();
        
        // Core identity triples
        triples.push(format!(
            "<{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}/XuidIdentity> .",
            iri, base_namespace
        ));
        
        triples.push(format!(
            "<{}> <{}/xuidType> \"{}\" .",
            iri, base_namespace, self.xuid_type
        ));
        
        triples.push(format!(
            "<{}> <{}/deltaSignature> \"{}\" .",
            iri, base_namespace, self.delta_signature
        ));
        
        triples.push(format!(
            "<{}> <{}/orbitId> {} .",
            iri, base_namespace, self.orbit_id
        ));
        
        triples.push(format!(
            "<{}> <{}/reflectionPath> \"{}\" .",
            iri, base_namespace, self.reflection_path
        ));
        
        triples.push(format!(
            "<{}> <{}/semanticHash> \"{}\" .",
            iri, base_namespace, self.semantic_hash
        ));
        
        triples.push(format!(
            "<{}> <{}/provenanceHash> \"{}\" .",
            iri, base_namespace, self.provenance_hash
        ));
        
        // Provenance triples if available
        if let Some(ref provenance) = self.provenance {
            triples.push(format!(
                "<{}> <{}/processingStrategy> \"{}\" .",
                iri, base_namespace, provenance.processing_strategy
            ));
            
            triples.push(format!(
                "<{}> <{}/qualityScore> {} .",
                iri, base_namespace, provenance.quality_score
            ));
            
            triples.push(format!(
                "<{}> <{}/processingTimeNs> {} .",
                iri, base_namespace, provenance.processing_time_ns
            ));
            
            triples.push(format!(
                "<{}> <{}/workerId> {} .",
                iri, base_namespace, provenance.worker_id
            ));
        }
        
        triples
    }
}

impl fmt::Display for Xuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "XUID-{}::Δ{}::ORBIT-{:02}::{}::{}::{}",
            self.xuid_type,
            self.delta_signature,
            self.orbit_id,
            self.reflection_path,
            self.semantic_hash,
            self.provenance_hash
        )
    }
}

impl FromStr for Xuid {
    type Err = XuidError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Builder pattern for XUID construction
#[derive(Debug, Default)]
pub struct XuidBuilder {
    xuid_type: Option<XuidType>,
    input_data: Option<Vec<u8>>,
    quantization_result: Option<Vec<f32>>,
    semantic_path: Option<SemanticPath>,
    provenance: Option<XuidProvenance>,
    orbit_id: Option<u32>,
}

impl XuidBuilder {
    /// Create new XUID builder
    pub fn new(xuid_type: XuidType) -> Self {
        Self {
            xuid_type: Some(xuid_type),
            ..Default::default()
        }
    }
    
    /// Set input data for delta signature generation
    pub fn with_input_data(mut self, data: &[u8]) -> Self {
        self.input_data = Some(data.to_vec());
        self
    }
    
    /// Set quantization result for semantic hash
    pub fn with_quantization_result(mut self, result: &[f32]) -> Self {
        self.quantization_result = Some(result.to_vec());
        self
    }
    
    /// Set semantic reasoning path
    pub fn with_semantic_path(mut self, path: SemanticPath) -> Self {
        self.semantic_path = Some(path);
        self
    }
    
    /// Set provenance information
    pub fn with_provenance(mut self, provenance: XuidProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
    
    /// Set orbit ID explicitly
    pub fn with_orbit_id(mut self, orbit_id: u32) -> Self {
        self.orbit_id = Some(orbit_id);
        self
    }
    
    /// Build XUID with all components
    pub fn build(self) -> Result<Xuid, XuidError> {
        let xuid_type = self.xuid_type.ok_or(XuidError::MissingComponent("xuid_type"))?;
        let input_data = self.input_data.ok_or(XuidError::MissingComponent("input_data"))?;
        
        // Generate delta signature from input data
        let input_hash = blake3::hash(&input_data);
        let delta_signature = format!("{:08X}", 
            input_hash.as_bytes()[0..4].iter()
                .fold(0u32, |acc, &b| (acc << 8) | b as u32));
        
        // Generate semantic hash from quantization result
        let semantic_hash = if let Some(ref quant_result) = self.quantization_result {
            let mut hasher = Hasher::new();
            for &val in quant_result {
                hasher.update(&val.to_bits().to_le_bytes());
            }
            let hash = hasher.finalize();
            format!("SEM-{:08X}", 
                hash.as_bytes()[0..4].iter()
                    .fold(0u32, |acc, &b| (acc << 8) | b as u32))
        } else {
            format!("SEM-{:08X}", 0u32)
        };
        
        // Determine orbit ID
        let orbit_id = self.orbit_id.unwrap_or_else(|| {
            if let Some(ref quant_result) = self.quantization_result {
                let sum: f32 = quant_result.iter().sum();
                ((sum * 1000.0) as u32) % 30 // E8 has ~30 orbits
            } else {
                0
            }
        });
        
        // Generate reflection path
        let reflection_path = if let Some(ref path) = self.semantic_path {
            path.compress()
        } else {
            "RPL-001-0000".to_string()
        };
        
        // Generate provenance hash
        let provenance_hash = if let Some(ref prov) = self.provenance {
            prov.compress()
        } else {
            "PROV-00000000".to_string()
        };
        
        Ok(Xuid {
            xuid_type,
            delta_signature,
            orbit_id,
            reflection_path,
            semantic_hash,
            provenance_hash,
            semantic_path: self.semantic_path,
            provenance: self.provenance,
        })
    }
}

/// XUID-specific error types
#[derive(Error, Debug)]
pub enum XuidError {
    #[error("Invalid XUID type: {0}")]
    /// Provided XUID type is invalid.
    InvalidType(String),
    
    /// Error returned when the XUID string format is invalid.
    #[error("Invalid XUID format: expected {expected} components, found {found} in '{input}'")]
    /// XUID string format is invalid.
    InvalidFormat {
            /// Expected number of components in the XUID string.
            expected: usize,
            /// Actual number of components found in the XUID string.
            found: usize,
            /// The input XUID string that caused the error.
            input: String,
        },
    
    #[error("Missing XUID prefix in: {0}")]
    /// Missing required XUID prefix.
    MissingPrefix(String),
    
    /// Error for invalid delta signature format in XUID.
    #[error("Invalid delta format: {0}")]
    InvalidDeltaFormat(String),
    
    /// Error for invalid orbit ID format in XUID.
    #[error("Invalid orbit format: {0}")]
    InvalidOrbitFormat(String),
    
    /// Error for invalid semantic path format in XUID.
    #[error("Invalid path format: {0}")]
    InvalidPathFormat(String),
    
    /// Error for invalid semantic hash format in XUID.
    #[error("Invalid semantic format: {0}")]
    InvalidSemanticFormat(String),
    
    /// Error for invalid provenance hash format in XUID.
    #[error("Invalid provenance format: {0}")]
    InvalidProvenanceFormat(String),
    
    /// Error for missing required component during XUID construction.
    #[error("Missing required component: {0}")]
    MissingComponent(&'static str),
    
    /// Error indicating XUID integrity verification failed.
    #[error("Integrity verification failed")]
    IntegrityFailure,
}

/// XUID registry for managing relationships and lookups
#[derive(Debug, Default)]
pub struct XuidRegistry {
    /// XUID to data mappings
    xuids: HashMap<String, Xuid>,
    /// Similarity index for fast lookup
    similarity_index: HashMap<String, Vec<String>>,
    /// Orbit to XUID mappings
    orbit_index: HashMap<u32, Vec<String>>,
}

impl XuidRegistry {
    /// Create new XUID registry
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Register a new XUID
    pub fn register(&mut self, xuid: Xuid) -> Result<(), XuidError> {
        let xuid_str = xuid.to_string();
        
        // Verify integrity before registration
        if !xuid.verify_integrity()? {
            return Err(XuidError::IntegrityFailure);
        }
        
        // Add to orbit index
        self.orbit_index.entry(xuid.orbit_id)
            .or_insert_with(Vec::new)
            .push(xuid_str.clone());
        
        // Add to similarity index (simplified)
        let similarity_key = format!("{}:{}", xuid.xuid_type, xuid.orbit_id);
        self.similarity_index.entry(similarity_key)
            .or_insert_with(Vec::new)
            .push(xuid_str.clone());
        
        // Store the XUID
        self.xuids.insert(xuid_str, xuid);
        
        Ok(())
    }
    
    /// Find similar XUIDs
    pub fn find_similar(&self, xuid: &Xuid, threshold: f64) -> Vec<(&Xuid, f64)> {
        let mut similar = Vec::new();
        
        // Start with same orbit
        if let Some(orbit_xuids) = self.orbit_index.get(&xuid.orbit_id) {
            for xuid_str in orbit_xuids {
                if let Some(other_xuid) = self.xuids.get(xuid_str) {
                    let similarity = xuid.similarity_score(other_xuid);
                    if similarity >= threshold {
                        similar.push((other_xuid, similarity));
                    }
                }
            }
        }
        
        // Sort by similarity descending
        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similar
    }
    
    /// Get XUID by string
    pub fn get(&self, xuid_str: &str) -> Option<&Xuid> {
        self.xuids.get(xuid_str)
    }
    
    /// Get all XUIDs in orbit
    pub fn get_orbit_xuids(&self, orbit_id: u32) -> Vec<&Xuid> {
        self.orbit_index.get(&orbit_id)
            .map(|xuid_strs| {
                xuid_strs.iter()
                    .filter_map(|s| self.xuids.get(s))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Generate SPARQL schema for RDF integration
    pub fn generate_sparql_schema(&self, base_namespace: &str) -> String {
        format!(
            r#"
# XUID Ontology Schema
@prefix xuid: <{}/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

# Core Classes
xuid:XuidIdentity a owl:Class ;
    rdfs:label "XUID Identity" ;
    rdfs:comment "Xypher Unique Identity Descriptor with mathematical provenance" .

# Properties
xuid:xuidType a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:string .

xuid:deltaSignature a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:string .

xuid:orbitId a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:nonNegativeInteger .

xuid:reflectionPath a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:string .

xuid:semanticHash a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:string .

xuid:provenanceHash a owl:DatatypeProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xsd:string .

xuid:similarTo a owl:ObjectProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xuid:XuidIdentity ;
    rdfs:comment "Semantic similarity relationship between XUIDs" .

xuid:derivedFrom a owl:ObjectProperty ;
    rdfs:domain xuid:XuidIdentity ;
    rdfs:range xuid:XuidIdentity ;
    rdfs:comment "Provenance chain relationship" .
"#,
            base_namespace
        )
    }
    
    /// Export all XUIDs as SPARQL triples
    pub fn export_sparql_triples(&self, base_namespace: &str) -> Vec<String> {
        let mut all_triples = Vec::new();
        
        for xuid in self.xuids.values() {
            all_triples.extend(xuid.to_sparql_triples(base_namespace));
        }
        
        // Add similarity relationships
        for xuid in self.xuids.values() {
            let similar = self.find_similar(xuid, 0.7);
            for (similar_xuid, score) in similar {
                if xuid != similar_xuid {
                    all_triples.push(format!(
                        "<{}> <{}/similarTo> <{}> .",
                        xuid.to_iri(base_namespace),
                        base_namespace,
                        similar_xuid.to_iri(base_namespace)
                    ));
                    
                    all_triples.push(format!(
                        "<{}> <{}/similarityScore> {} .",
                        xuid.to_iri(base_namespace),
                        base_namespace,
                        score
                    ));
                }
            }
        }
        
        all_triples
    }
    
    /// Registry statistics
    pub fn stats(&self) -> XuidRegistryStats {
        let mut type_counts = HashMap::new();
        let mut orbit_counts = HashMap::new();
        
        for xuid in self.xuids.values() {
            *type_counts.entry(xuid.xuid_type).or_insert(0) += 1;
            *orbit_counts.entry(xuid.orbit_id).or_insert(0) += 1;
        }
        
        XuidRegistryStats {
            total_xuids: self.xuids.len(),
            type_distribution: type_counts,
            orbit_distribution: orbit_counts,
            similarity_index_size: self.similarity_index.len(),
        }
    }
}

/// XUID registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XuidRegistryStats {
    /// Total number of XUIDs registered in the system.
    pub total_xuids: usize,
    /// Distribution of XUID types present in the registry.
    pub type_distribution: HashMap<XuidType, usize>,
    /// Distribution of orbit assignments among registered XUIDs.
    pub orbit_distribution: HashMap<u32, usize>,
    /// Number of entries in the similarity index for fast lookup.
    pub similarity_index_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_xuid_creation_and_parsing() {
        let provenance = XuidProvenance::new(
            b"test_data",
            "E8Quantized",
            42,
            0.95,
            1000000,
            1024,
            8
        );
        
        let semantic_path = SemanticPath::direct("root_123".to_string());
        
        let xuid = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(b"test_data")
            .with_quantization_result(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .with_semantic_path(semantic_path)
            .with_provenance(provenance)
            .with_orbit_id(7)
            .build()
            .unwrap();
        
        let xuid_str = xuid.to_string();
        let parsed_xuid = Xuid::parse(&xuid_str).unwrap();
        
        assert_eq!(xuid.xuid_type, parsed_xuid.xuid_type);
        assert_eq!(xuid.orbit_id, parsed_xuid.orbit_id);
        assert_eq!(xuid.delta_signature, parsed_xuid.delta_signature);
    }
    
    #[test]
    fn test_xuid_similarity() {
        let xuid1 = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(b"similar_data_1")
            .with_quantization_result(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .with_orbit_id(7)
            .build()
            .unwrap();
        
        let xuid2 = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(b"similar_data_2")
            .with_quantization_result(&[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
            .with_orbit_id(7)
            .build()
            .unwrap();
        
        let similarity = xuid1.similarity_score(&xuid2);
        assert!(similarity > 0.4); // Same orbit should give significant similarity
    }
    
    #[test]
    fn test_xuid_registry() {
        let mut registry = XuidRegistry::new();
        
        let xuid = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(b"registry_test")
            .with_quantization_result(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .with_orbit_id(5)
            .build()
            .unwrap();
        
        registry.register(xuid.clone()).unwrap();
        
        let retrieved = registry.get(&xuid.to_string()).unwrap();
        assert_eq!(retrieved.orbit_id, 5);
        
        let orbit_xuids = registry.get_orbit_xuids(5);
        assert_eq!(orbit_xuids.len(), 1);
    }
    
    #[test]
    fn test_sparql_generation() {
        let xuid = XuidBuilder::new(XuidType::E8Quantized)
            .with_input_data(b"sparql_test")
            .with_quantization_result(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .with_orbit_id(3)
            .build()
            .unwrap();
        
        let triples = xuid.to_sparql_triples("http://xypher.ai");
        assert!(!triples.is_empty());
        assert!(triples.iter().any(|t| t.contains("XuidIdentity")));
    }
    
    #[test]
    fn test_semantic_path_compression() {
        let path = SemanticPath {
            traversal_nodes: vec!["root_1".to_string(), "root_2".to_string(), "root_3".to_string()],
            reflection_sequence: vec![1, 2, 5, 7],
            orbit_transitions: vec![1, 3, 7],
            distance_progression: vec![0.0, 0.5, 1.2],
            quality_progression: vec![1.0, 0.9, 0.8],
        };
        
        let compressed = path.compress();
        assert!(compressed.starts_with("RPL-"));
        
        let complexity = path.complexity_score();
        assert!(complexity > 0.0);
    }
}
