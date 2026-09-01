use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use clap::{Parser, ValueEnum};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::Value;

use crate::{Error, Result};

const DEFAULT_BOOK_URL: &str = "https://www.gutenberg.org/files/1661/1661-0.txt";

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LatencyMode {
    Api,
    Generation,
    None,
}

impl std::fmt::Display for LatencyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Api => "api",
            Self::Generation => "generation",
            Self::None => "none",
        })
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Md,
    Json,
    Csv,
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Benchmark OpenAI-compatible chat completion endpoints"
)]
pub struct Args {
    /// Base API URL, normally ending in /v1.
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub base_url: String,

    /// API key. OPENAI_API_KEY is also accepted.
    #[arg(
        long,
        env = "OPENAI_API_KEY",
        default_value = "EMPTY",
        hide_env_values = true
    )]
    pub api_key: String,

    /// Model shown in reports. Auto-detected when the endpoint exposes exactly one model.
    #[arg(long)]
    pub model: Option<String>,

    /// Model identifier sent to the endpoint. Defaults to --model or the detected ID.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Prompt-processing token counts.
    #[arg(long = "pp", num_args = 1.., value_delimiter = ',', default_values_t = [2048])]
    pub pp_counts: Vec<usize>,

    /// Generated-token counts.
    #[arg(long = "tg", num_args = 1.., value_delimiter = ',', default_values_t = [32])]
    pub tg_counts: Vec<usize>,

    /// Context depths in tokens.
    #[arg(long = "depth", num_args = 1.., value_delimiter = ',', default_values_t = [0])]
    pub depths: Vec<usize>,

    /// Concurrent requests per measured batch.
    #[arg(long = "concurrency", num_args = 1.., value_delimiter = ',', default_values_t = [1])]
    pub concurrency_levels: Vec<usize>,

    /// Measured batches per test shape.
    #[arg(long = "runs", default_value_t = 3)]
    pub num_runs: usize,

    /// Discarded batches per test shape.
    #[arg(long, default_value_t = 1)]
    pub warmup_runs: usize,

    /// Add server-specific fields that force exactly --tg output tokens.
    #[arg(long)]
    pub exact_tg: bool,

    /// Make prompts unique and ask compatible servers not to cache them.
    #[arg(long)]
    pub no_cache: bool,

    /// Preload context separately before each depth > 0 request.
    #[arg(long)]
    pub enable_prefix_caching: bool,

    /// Baseline subtracted from time-to-first-response.
    #[arg(long, value_enum, default_value_t = LatencyMode::Api)]
    pub latency_mode: LatencyMode,

    /// Skip endpoint token calibration and assume four characters per token.
    #[arg(long)]
    pub no_warmup: bool,

    /// Do not compensate target prompt sizes for measured chat-template overhead.
    #[arg(long)]
    pub no_adapt_prompt: bool,

    /// Skip the simple endpoint/model coherence check.
    #[arg(long)]
    pub skip_coherence: bool,

    /// Local UTF-8 corpus. Avoids downloading --book-url.
    #[arg(long, conflicts_with = "book_url")]
    pub prompt_file: Option<PathBuf>,

    /// Text corpus URL.
    #[arg(long, default_value = DEFAULT_BOOK_URL)]
    pub book_url: String,

    /// Run a shell command after every batch.
    #[arg(long)]
    pub post_run_cmd: Option<String>,

    /// Extra request JSON: key=value,key2=true. May be repeated.
    #[arg(long, action = clap::ArgAction::Append)]
    pub extra_body: Vec<String>,

    /// Stop after the first failed request.
    #[arg(long)]
    pub exit_on_first_fail: bool,

    /// Save/print no report if any request fails; implies --exit-on-first-fail.
    #[arg(long)]
    pub no_results_on_fail: bool,

    /// Report destination. Prints to stdout when omitted.
    #[arg(long)]
    pub save_result: Option<PathBuf>,

    /// Report format.
    #[arg(long = "format", value_enum, default_value_t = OutputFormat::Md)]
    pub output_format: OutputFormat,

    /// Total request timeout in seconds.
    #[arg(long, default_value_t = 3600)]
    pub timeout: u64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub served_model_name: String,
    pub pp_counts: Vec<usize>,
    pub tg_counts: Vec<usize>,
    pub depths: Vec<usize>,
    pub concurrency_levels: Vec<usize>,
    pub num_runs: usize,
    pub warmup_runs: usize,
    pub exact_tg: bool,
    pub no_cache: bool,
    pub enable_prefix_caching: bool,
    pub latency_mode: LatencyMode,
    pub no_warmup: bool,
    pub adapt_prompt: bool,
    pub skip_coherence: bool,
    pub prompt_file: Option<PathBuf>,
    pub book_url: String,
    pub post_run_cmd: Option<String>,
    pub extra_body: BTreeMap<String, Value>,
    pub exit_on_first_fail: bool,
    pub no_results_on_fail: bool,
    pub save_result: Option<PathBuf>,
    pub output_format: OutputFormat,
    pub timeout: Duration,
}

#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    root: String,
}

impl BenchmarkConfig {
    pub async fn resolve(args: Args) -> Result<Self> {
        validate_positive("--pp", &args.pp_counts)?;
        validate_positive("--tg", &args.tg_counts)?;
        validate_positive("--concurrency", &args.concurrency_levels)?;
        if args.num_runs == 0 {
            return Err(Error::Config("--runs must be greater than zero".into()));
        }

        let base_url = args.base_url.trim_end_matches('/').to_owned();
        if base_url.is_empty() {
            return Err(Error::Config("--base-url cannot be empty".into()));
        }
        let (model, detected_served) = match args.model {
            Some(model) => (model.clone(), model),
            None => detect_model(&base_url, &args.api_key, args.timeout).await?,
        };
        let served_model_name = args.served_model_name.unwrap_or(detected_served);
        Ok(Self {
            base_url,
            api_key: args.api_key,
            model,
            served_model_name,
            pp_counts: args.pp_counts,
            tg_counts: args.tg_counts,
            depths: args.depths,
            concurrency_levels: args.concurrency_levels,
            num_runs: args.num_runs,
            warmup_runs: args.warmup_runs,
            exact_tg: args.exact_tg,
            no_cache: args.no_cache,
            enable_prefix_caching: args.enable_prefix_caching,
            latency_mode: args.latency_mode,
            no_warmup: args.no_warmup,
            adapt_prompt: !args.no_adapt_prompt,
            skip_coherence: args.skip_coherence,
            prompt_file: args.prompt_file,
            book_url: args.book_url,
            post_run_cmd: args.post_run_cmd,
            extra_body: parse_extra_body(&args.extra_body)?,
            exit_on_first_fail: args.exit_on_first_fail || args.no_results_on_fail,
            no_results_on_fail: args.no_results_on_fail,
            save_result: args.save_result,
            output_format: args.output_format,
            timeout: Duration::from_secs(args.timeout),
        })
    }
}

fn validate_positive(name: &str, values: &[usize]) -> Result<()> {
    if values.is_empty() || values.contains(&0) {
        return Err(Error::Config(format!(
            "{name} values must be greater than zero"
        )));
    }
    Ok(())
}

pub fn parse_extra_body(items: &[String]) -> Result<BTreeMap<String, Value>> {
    let mut result = BTreeMap::new();
    for entry in items.iter().flat_map(|item| item.split(',')) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (key, raw) = entry
            .split_once('=')
            .or_else(|| entry.split_once(':'))
            .ok_or_else(|| {
                Error::Config(format!(
                    "invalid --extra-body entry '{entry}'; expected key=value"
                ))
            })?;
        let key = key.trim();
        if key.is_empty() {
            return Err(Error::Config("--extra-body contains an empty key".into()));
        }
        let raw = raw.trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()));
        result.insert(key.to_owned(), value);
    }
    Ok(result)
}

async fn detect_model(base_url: &str, api_key: &str, timeout: u64) -> Result<(String, String)> {
    eprintln!("No model specified; querying {base_url}/models ...");
    let url = format!("{base_url}/models");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout.min(30)))
        .build()
        .map_err(|source| Error::Request {
            url: url.clone(),
            source,
        })?;
    let mut headers = HeaderMap::new();
    if !api_key.is_empty() && api_key != "EMPTY" {
        let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| Error::Config("API key contains invalid header characters".into()))?;
        headers.insert(AUTHORIZATION, value);
    }
    let response = client
        .get(&url)
        .headers(headers)
        .send()
        .await
        .map_err(|source| Error::Request {
            url: url.clone(),
            source,
        })?;
    let status = response.status();
    let body = response.text().await.map_err(|source| Error::Request {
        url: url.clone(),
        source,
    })?;
    if !status.is_success() {
        return Err(Error::HttpStatus { url, status, body });
    }
    let parsed: ModelsResponse = serde_json::from_str(&body).map_err(|source| Error::Json {
        context: "GET /models".into(),
        source,
    })?;
    let entries = if parsed.data.is_empty() {
        parsed.models
    } else {
        parsed.data
    };
    if entries.len() != 1 {
        let names = entries.iter().map(model_id).collect::<Vec<_>>().join(", ");
        return Err(Error::Config(format!(
            "model auto-detection requires exactly one endpoint model (found {}: {}); pass --model",
            entries.len(),
            names
        )));
    }
    let entry = &entries[0];
    let served = model_id(entry);
    if served.is_empty() {
        return Err(Error::Protocol {
            endpoint: url,
            message: "model entry has no id".into(),
        });
    }
    let report_model = if entry.root.contains('/') {
        entry.root.clone()
    } else {
        served.clone()
    };
    Ok((report_model, served))
}

fn model_id(entry: &ModelEntry) -> String {
    if !entry.id.is_empty() {
        entry.id.clone()
    } else {
        entry.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_body_accepts_json_and_strings() {
        let parsed =
            parse_extra_body(&["min_tokens=12,ignore_eos:true".into(), "mode=fast".into()])
                .unwrap();
        assert_eq!(parsed["min_tokens"], 12);
        assert_eq!(parsed["ignore_eos"], true);
        assert_eq!(parsed["mode"], "fast");
    }

    #[test]
    fn extra_body_rejects_missing_separator() {
        assert!(parse_extra_body(&["ignore_eos".into()]).is_err());
    }
}
