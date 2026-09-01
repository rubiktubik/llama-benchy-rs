use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("request to {url} failed: {source}")]
    Request {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("endpoint returned HTTP {status} for {url}: {body}")]
    HttpStatus {
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("invalid response from {endpoint}: {message}")]
    Protocol { endpoint: String, message: String },

    #[error("failed to parse JSON from {context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to load tokenizer '{name}': {message}")]
    Tokenizer { name: String, message: String },

    #[error("failed to read {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to serialize CSV report: {0}")]
    Csv(#[from] csv::Error),

    #[error("post-run command failed with status {status}: {command}")]
    PostRun { command: String, status: String },

    #[error("failed to start post-run command '{command}': {source}")]
    StartPostRun {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("benchmark request failed: {0}")]
    Benchmark(String),

    #[error("benchmark was interrupted")]
    Interrupted,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
