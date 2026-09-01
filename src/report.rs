use std::{io::IsTerminal, path::Path};

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
    pub measured_runs: usize,
    pub benchmarks: Vec<BenchmarkRun>,
}

impl BenchmarkReport {
    pub async fn write(
        &self,
        path: Option<&Path>,
        format: OutputFormat,
        detailed: bool,
    ) -> Result<()> {
        let color = path.is_none()
            && std::io::stdout().is_terminal()
            && std::env::var_os("NO_COLOR").is_none();
        let output = match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).map_err(|source| Error::Json {
                    context: "benchmark report".into(),
                    source,
                })?
            }
            OutputFormat::Md if detailed => self.detailed_markdown(),
            OutputFormat::Md => self.compact_markdown(),
            OutputFormat::Pretty if detailed => self.detailed_markdown(),
            OutputFormat::Pretty => self.pretty(color),
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

    fn pretty(&self, color: bool) -> String {
        let title = paint("rust-benchy · benchmark summary", "1;36", color);
        let model = paint(&self.model, "1", color);
        let latency = format!("{} · {:.1} ms", self.latency_mode, self.latency_ms);
        let mut output = format!(
            "{title}\n\n  Model      {model}\n  Runs       {} per shape\n  Latency    {latency}\n\n",
            self.measured_runs,
        );

        let show_phase = self.prefix_caching_enabled;
        let mut headers = Vec::new();
        if show_phase {
            headers.push("Phase");
        }
        headers.extend([
            "Prompt",
            "Output",
            "Context",
            "Load",
            "Input tok/s",
            "Output tok/s",
            "TTFT",
        ]);
        let mut rows = Vec::with_capacity(self.benchmarks.len());
        for run in &self.benchmarks {
            let mut row = Vec::new();
            if show_phase {
                row.push(
                    if run.is_context_prefill_phase {
                        "context load"
                    } else {
                        "inference"
                    }
                    .to_owned(),
                );
            }
            row.extend([
                if run.is_context_prefill_phase {
                    "—".to_owned()
                } else {
                    grouped_integer(run.prompt_size)
                },
                grouped_integer(run.response_size),
                grouped_integer(run.context_size),
                format!("{}×", run.concurrency),
                compact_rate(run.pp_throughput.as_ref()),
                compact_rate(run.tg_throughput.as_ref()),
                compact_duration(run.e2e_ttft.as_ref()),
            ]);
            rows.push(row);
        }
        output.push_str(&unicode_table(&headers, &rows, show_phase, color));
        output.push_str("\n  Input = prompt processing · Output = token generation · TTFT = time to first token\n");
        output.push_str(
            "  — no measurable token interval (the endpoint likely buffered its output)\n",
        );
        if self.failed_requests > 0 {
            output.push_str(&format!(
                "\n  {}\n",
                paint(
                    &format!("⚠ {} request(s) failed", self.failed_requests),
                    "1;31",
                    color,
                )
            ));
        }
        output
    }

    fn compact_markdown(&self) -> String {
        let show_phase = self.prefix_caching_enabled;
        let mut output = format!(
            "# Benchmark results\n\n**Model:** {}  \n**Runs per shape:** {}  \n**Latency baseline:** {} ({:.1} ms)\n\n",
            self.model, self.measured_runs, self.latency_mode, self.latency_ms,
        );
        if show_phase {
            output.push_str(
                "| Phase | Prompt | Output | Context | Concurrent | Input tok/s | Output tok/s | TTFT |\n|:--|--:|--:|--:|--:|--:|--:|--:|\n",
            );
        } else {
            output.push_str(
                "| Prompt | Output | Context | Concurrent | Input tok/s | Output tok/s | TTFT |\n|--:|--:|--:|--:|--:|--:|--:|\n",
            );
        }

        for run in &self.benchmarks {
            let prompt = if run.is_context_prefill_phase {
                "—".to_owned()
            } else {
                grouped_integer(run.prompt_size)
            };
            let output_tokens = grouped_integer(run.response_size);
            let context = grouped_integer(run.context_size);
            let input_rate = compact_rate(run.pp_throughput.as_ref());
            let output_rate = compact_rate(run.tg_throughput.as_ref());
            let ttft = compact_duration(run.e2e_ttft.as_ref());

            if show_phase {
                let phase = if run.is_context_prefill_phase {
                    "context load"
                } else {
                    "inference"
                };
                output.push_str(&format!(
                    "| {phase} | {prompt} | {output_tokens} | {context} | {} | {input_rate} | {output_rate} | {ttft} |\n",
                    run.concurrency,
                ));
            } else {
                output.push_str(&format!(
                    "| {prompt} | {output_tokens} | {context} | {} | {input_rate} | {output_rate} | {ttft} |\n",
                    run.concurrency,
                ));
            }
        }

        output.push_str(
            "\nValues are means. `—` means no measurable token interval was observed, usually because output was buffered. Use `--detailed` for variance, peak throughput, TTFR, and per-request values.\n",
        );
        if self.failed_requests > 0 {
            output.push_str(&format!(
                "\n**Warning:** {} request(s) failed.\n",
                self.failed_requests
            ));
        }
        output
    }

    fn detailed_markdown(&self) -> String {
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

fn grouped_integer(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn compact_rate(metric: Option<&BenchmarkMetric>) -> String {
    let Some(value) = metric.map(|metric| metric.mean) else {
        return "—".into();
    };
    if value >= 100.0 {
        grouped_integer(value.round() as usize)
    } else {
        format!("{value:.1}")
    }
}

fn compact_duration(metric: Option<&BenchmarkMetric>) -> String {
    let Some(milliseconds) = metric.map(|metric| metric.mean) else {
        return "—".into();
    };
    if milliseconds >= 1000.0 {
        format!("{:.2} s", milliseconds / 1000.0)
    } else {
        format!("{milliseconds:.0} ms")
    }
}

fn unicode_table(headers: &[&str], rows: &[Vec<String>], has_phase: bool, color: bool) -> String {
    let mut widths = headers
        .iter()
        .map(|header| header.chars().count())
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let border = |left: char, middle: char, right: char| {
        let segments = widths
            .iter()
            .map(|width| "─".repeat(width + 2))
            .collect::<Vec<_>>()
            .join(&middle.to_string());
        format!("{left}{segments}{right}\n")
    };
    let header_cells = headers
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    let mut output = border('┌', '┬', '┐');
    output.push_str(&pretty_row(&header_cells, &widths, has_phase, true, color));
    output.push_str(&border('├', '┼', '┤'));
    for row in rows {
        output.push_str(&pretty_row(row, &widths, has_phase, false, color));
    }
    output.push_str(&border('└', '┴', '┘'));
    output
}

fn pretty_row(
    cells: &[String],
    widths: &[usize],
    has_phase: bool,
    header: bool,
    color: bool,
) -> String {
    let input_column = if has_phase { 5 } else { 4 };
    let output_column = input_column + 1;
    let ttft_column = output_column + 1;
    let rendered = cells
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let padding = widths[index] - cell.chars().count();
            let aligned = if header || (has_phase && index == 0) {
                format!("{cell}{}", " ".repeat(padding))
            } else {
                format!("{}{cell}", " ".repeat(padding))
            };
            let code = if header {
                Some("1;36")
            } else if index == input_column {
                Some("1;32")
            } else if index == output_column {
                Some("1;35")
            } else if index == ttft_column {
                Some("1;33")
            } else {
                None
            };
            code.map_or(aligned.clone(), |code| paint(&aligned, code, color))
        })
        .collect::<Vec<_>>()
        .join(" │ ");
    format!("│ {rendered} │\n")
}

fn paint(text: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(mean: f64, std: f64) -> BenchmarkMetric {
        BenchmarkMetric {
            mean,
            std,
            values: vec![mean],
        }
    }

    fn report() -> BenchmarkReport {
        BenchmarkReport {
            version: "0.1.0".into(),
            timestamp: "2026-09-01T12:00:00Z".into(),
            latency_mode: "generation".into(),
            latency_ms: 12.3,
            model: "test-model".into(),
            prefix_caching_enabled: false,
            max_concurrency: 4,
            failed_requests: 0,
            measured_runs: 3,
            benchmarks: vec![BenchmarkRun {
                concurrency: 4,
                context_size: 4096,
                prompt_size: 2048,
                response_size: 128,
                is_context_prefill_phase: false,
                pp_throughput: Some(metric(2287.89, 38.26)),
                pp_req_throughput: Some(metric(571.97, 12.0)),
                tg_throughput: Some(metric(67.95, 3.93)),
                tg_req_throughput: Some(metric(16.98, 1.0)),
                peak_throughput: Some(metric(80.0, 2.0)),
                peak_req_throughput: Some(metric(20.0, 1.0)),
                ttfr: Some(metric(2770.77, 45.26)),
                est_ppt: Some(metric(2758.47, 45.26)),
                e2e_ttft: Some(metric(2770.79, 45.26)),
            }],
        }
    }

    #[test]
    fn pretty_report_prioritizes_key_metrics() {
        let output = report().pretty(false);
        assert!(output.contains("rust-benchy · benchmark summary"));
        assert!(output.contains(
            "│  2,048 │    128 │   4,096 │   4× │       2,288 │         68.0 │ 2.77 s │"
        ));
        assert!(!output.contains('±'));
        assert!(!output.contains("test-model |"));
    }

    #[test]
    fn detailed_report_keeps_variance() {
        let output = report().detailed_markdown();
        assert!(output.contains("2287.89 ± 38.26"));
        assert!(output.contains("t/s (req)"));
    }
}
