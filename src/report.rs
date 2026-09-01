use std::path::Path;

use serde::Serialize;

use crate::{
    Error, Result,
    config::OutputFormat,
    metrics::{BenchmarkMetric, BenchmarkRun},
};

#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub version: String,
    pub timestamp: String,
    pub latency_mode: String,
    pub latency_ms: f64,
    pub model: String,
    pub prefix_caching_enabled: bool,
    pub max_concurrency: usize,
    pub failed_requests: usize,
    pub benchmarks: Vec<BenchmarkRun>,
}

impl BenchmarkReport {
    pub async fn write(&self, path: Option<&Path>, format: OutputFormat) -> Result<()> {
        let output = match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).map_err(|source| Error::Json {
                    context: "benchmark report".into(),
                    source,
                })?
            }
            OutputFormat::Md => self.markdown(),
            OutputFormat::Csv => self.csv()?,
        };
        if let Some(path) = path {
            tokio::fs::write(path, output)
                .await
                .map_err(|source| Error::WriteFile {
                    path: path.to_owned(),
                    source,
                })?;
            eprintln!("Saved report to {}", path.display());
        } else {
            println!("{output}");
        }
        Ok(())
    }

    fn markdown(&self) -> String {
        let many = self.max_concurrency > 1;
        let mut output = if many {
            "| model | test | t/s (total) | t/s (req) | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |\n|:--|--:|--:|--:|--:|--:|--:|--:|\n".to_owned()
        } else {
            "| model | test | t/s | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |\n|:--|--:|--:|--:|--:|--:|--:|\n".to_owned()
        };
        for run in &self.benchmarks {
            let suffix = if self.max_concurrency > 1 {
                format!(" (c{})", run.concurrency)
            } else {
                String::new()
            };
            let depth = if run.context_size > 0 {
                format!(" @ d{}", run.context_size)
            } else {
                String::new()
            };
            let pp_name = if run.is_context_prefill_phase {
                format!("ctx_pp @ d{}{}", run.context_size, suffix)
            } else {
                format!("pp{}{}{}", run.prompt_size, depth, suffix)
            };
            push_row(
                &mut output,
                many,
                &self.model,
                &pp_name,
                run.pp_throughput.as_ref(),
                run.pp_req_throughput.as_ref(),
                None,
                run.ttfr.as_ref(),
                run.est_ppt.as_ref(),
                run.e2e_ttft.as_ref(),
            );
            if run.tg_throughput.is_some() || run.peak_throughput.is_some() {
                let tg_name = if run.is_context_prefill_phase {
                    format!("ctx_tg @ d{}{}", run.context_size, suffix)
                } else {
                    format!("tg{}{}{}", run.response_size, depth, suffix)
                };
                push_row(
                    &mut output,
                    many,
                    &self.model,
                    &tg_name,
                    run.tg_throughput.as_ref(),
                    run.tg_req_throughput.as_ref(),
                    run.peak_throughput.as_ref(),
                    None,
                    None,
                    None,
                );
            }
        }
        output.push_str(&format!(
            "\nrust-benchy {} | {} | latency: {} ({:.2} ms) | failed requests: {}\n",
            self.version, self.timestamp, self.latency_mode, self.latency_ms, self.failed_requests,
        ));
        output
    }

    fn csv(&self) -> Result<String> {
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record([
            "model",
            "test",
            "concurrency",
            "t_s_mean",
            "t_s_std",
            "t_s_req_mean",
            "t_s_req_std",
            "peak_t_s_mean",
            "peak_t_s_std",
            "ttfr_ms_mean",
            "ttfr_ms_std",
            "est_ppt_ms_mean",
            "est_ppt_ms_std",
            "e2e_ttft_ms_mean",
            "e2e_ttft_ms_std",
        ])?;
        for run in &self.benchmarks {
            let depth = if run.context_size > 0 {
                format!(" @ d{}", run.context_size)
            } else {
                String::new()
            };
            let pp_name = if run.is_context_prefill_phase {
                format!("ctx_pp @ d{}", run.context_size)
            } else {
                format!("pp{}{}", run.prompt_size, depth)
            };
            writer.write_record(csv_record(
                &self.model,
                &pp_name,
                run.concurrency,
                run.pp_throughput.as_ref(),
                run.pp_req_throughput.as_ref(),
                None,
                run.ttfr.as_ref(),
                run.est_ppt.as_ref(),
                run.e2e_ttft.as_ref(),
            ))?;
            if run.tg_throughput.is_some() || run.peak_throughput.is_some() {
                let tg_name = if run.is_context_prefill_phase {
                    format!("ctx_tg @ d{}", run.context_size)
                } else {
                    format!("tg{}{}", run.response_size, depth)
                };
                writer.write_record(csv_record(
                    &self.model,
                    &tg_name,
                    run.concurrency,
                    run.tg_throughput.as_ref(),
                    run.tg_req_throughput.as_ref(),
                    run.peak_throughput.as_ref(),
                    None,
                    None,
                    None,
                ))?;
            }
        }
        writer.flush().map_err(|source| Error::WriteFile {
            path: "<CSV buffer>".into(),
            source,
        })?;
        String::from_utf8(writer.into_inner().map_err(|error| Error::WriteFile {
            path: "<CSV buffer>".into(),
            source: error.into_error(),
        })?)
        .map_err(|error| Error::Config(format!("CSV report was not UTF-8: {error}")))
    }
}

fn fmt(metric: Option<&BenchmarkMetric>) -> String {
    metric
        .map(|metric| format!("{:.2} ± {:.2}", metric.mean, metric.std))
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn push_row(
    output: &mut String,
    many: bool,
    model: &str,
    test: &str,
    rate: Option<&BenchmarkMetric>,
    req_rate: Option<&BenchmarkMetric>,
    peak: Option<&BenchmarkMetric>,
    ttfr: Option<&BenchmarkMetric>,
    ppt: Option<&BenchmarkMetric>,
    ttft: Option<&BenchmarkMetric>,
) {
    if many {
        output.push_str(&format!(
            "| {model} | {test} | {} | {} | {} | {} | {} | {} |\n",
            fmt(rate),
            fmt(req_rate),
            fmt(peak),
            fmt(ttfr),
            fmt(ppt),
            fmt(ttft)
        ));
    } else {
        output.push_str(&format!(
            "| {model} | {test} | {} | {} | {} | {} | {} |\n",
            fmt(rate),
            fmt(peak),
            fmt(ttfr),
            fmt(ppt),
            fmt(ttft)
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn csv_record(
    model: &str,
    test: &str,
    concurrency: usize,
    rate: Option<&BenchmarkMetric>,
    req_rate: Option<&BenchmarkMetric>,
    peak: Option<&BenchmarkMetric>,
    ttfr: Option<&BenchmarkMetric>,
    ppt: Option<&BenchmarkMetric>,
    ttft: Option<&BenchmarkMetric>,
) -> Vec<String> {
    let pair = |metric: Option<&BenchmarkMetric>| {
        metric
            .map(|m| (m.mean.to_string(), m.std.to_string()))
            .unwrap_or_default()
    };
    let (rate_mean, rate_std) = pair(rate);
    let (req_mean, req_std) = pair(req_rate);
    let (peak_mean, peak_std) = pair(peak);
    let (ttfr_mean, ttfr_std) = pair(ttfr);
    let (ppt_mean, ppt_std) = pair(ppt);
    let (ttft_mean, ttft_std) = pair(ttft);
    vec![
        model.into(),
        test.into(),
        concurrency.to_string(),
        rate_mean,
        rate_std,
        req_mean,
        req_std,
        peak_mean,
        peak_std,
        ttfr_mean,
        ttfr_std,
        ppt_mean,
        ppt_std,
        ttft_mean,
        ttft_std,
    ]
}
