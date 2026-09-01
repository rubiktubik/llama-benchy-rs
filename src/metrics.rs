use serde::Serialize;

use crate::client::RequestResult;

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkMetric {
    pub mean: f64,
    pub std: f64,
    pub values: Vec<f64>,
}

impl BenchmarkMetric {
    fn new(values: Vec<f64>, multiplier: f64) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        Some(Self {
            mean: mean * multiplier,
            std: variance.sqrt() * multiplier,
            values: values.into_iter().map(|value| value * multiplier).collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRun {
    pub concurrency: usize,
    pub context_size: usize,
    pub prompt_size: usize,
    pub response_size: usize,
    pub is_context_prefill_phase: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pp_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pp_req_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tg_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tg_req_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_req_throughput: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttfr: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub est_ppt: Option<BenchmarkMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub e2e_ttft: Option<BenchmarkMetric>,
}

#[derive(Default)]
struct Aggregates {
    pp_req: Vec<f64>,
    tg_req: Vec<f64>,
    pp_batch: Vec<f64>,
    tg_batch: Vec<f64>,
    peak_batch: Vec<f64>,
    peak_req: Vec<f64>,
    ttfr: Vec<f64>,
    est_ppt: Vec<f64>,
    e2e_ttft: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkShape {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub context_tokens: usize,
    pub concurrency: usize,
    pub expected_prompt_tokens: usize,
    pub is_context_phase: bool,
}

pub fn aggregate(
    shape: BenchmarkShape,
    batches: &[Vec<RequestResult>],
    latency: f64,
) -> BenchmarkRun {
    let mut aggregate = Aggregates::default();
    for batch in batches {
        process_batch(batch, latency, shape.expected_prompt_tokens, &mut aggregate);
    }
    let pp_values = if shape.concurrency > 1 {
        aggregate.pp_batch
    } else {
        aggregate.pp_req.clone()
    };
    let tg_values = if shape.concurrency > 1 {
        aggregate.tg_batch
    } else {
        aggregate.tg_req.clone()
    };
    BenchmarkRun {
        concurrency: shape.concurrency,
        context_size: shape.context_tokens,
        prompt_size: shape.prompt_tokens,
        response_size: shape.generated_tokens,
        is_context_prefill_phase: shape.is_context_phase,
        pp_throughput: BenchmarkMetric::new(pp_values, 1.0),
        pp_req_throughput: BenchmarkMetric::new(aggregate.pp_req, 1.0),
        tg_throughput: BenchmarkMetric::new(tg_values, 1.0),
        tg_req_throughput: BenchmarkMetric::new(aggregate.tg_req, 1.0),
        peak_throughput: BenchmarkMetric::new(aggregate.peak_batch, 1.0),
        peak_req_throughput: BenchmarkMetric::new(aggregate.peak_req, 1.0),
        ttfr: BenchmarkMetric::new(aggregate.ttfr, 1000.0),
        est_ppt: BenchmarkMetric::new(aggregate.est_ppt, 1000.0),
        e2e_ttft: BenchmarkMetric::new(aggregate.e2e_ttft, 1000.0),
    }
}

fn process_batch(
    results: &[RequestResult],
    latency: f64,
    expected_tokens: usize,
    aggregate: &mut Aggregates,
) {
    if results.is_empty() {
        return;
    }
    let mut prompt_tokens_total = 0usize;
    let mut all_timestamps = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut first_tokens = Vec::new();
    let mut last_tokens = Vec::new();

    for result in results {
        starts.push(result.start_ts);
        ends.push(result.end_ts);
        all_timestamps.extend_from_slice(&result.token_timestamps);
        if let Some(last) = result.token_timestamps.last() {
            last_tokens.push(*last);
        } else {
            last_tokens.push(result.end_ts);
        }
        // Usage describes the entire templated request. During a prefix-cache
        // inference phase, however, this metric intentionally covers only the
        // new prompt. Accept usage when it plausibly describes this phase and
        // otherwise keep the calibrated target, matching llama-bench semantics.
        let reported_difference = result.prompt_tokens.abs_diff(expected_tokens);
        let prompt_tokens = if result.prompt_tokens > 0
            && reported_difference.saturating_mul(5) < expected_tokens
        {
            result.prompt_tokens
        } else {
            expected_tokens
        };
        prompt_tokens_total += prompt_tokens;

        if let Some(ttfr_at) = result.first_response_ts {
            let ttfr = (ttfr_at - result.start_ts).max(0.0);
            aggregate.ttfr.push(ttfr);
            let est_ppt = (ttfr - latency).max(0.0);
            aggregate.est_ppt.push(est_ppt);
            if est_ppt > 0.0 {
                aggregate.pp_req.push(prompt_tokens as f64 / est_ppt);
            }
        }
        if let Some(first) = result.first_token_ts {
            first_tokens.push(first);
            aggregate.e2e_ttft.push((first - result.start_ts).max(0.0));
        }
        if let (Some(first), Some(last)) = (
            result.token_timestamps.first(),
            result.token_timestamps.last(),
        ) {
            let observed = count_after_first(&result.token_timestamps);
            let duration = last - first;
            if observed > 0 && duration > 0.0 {
                aggregate.tg_req.push(observed as f64 / duration);
            }
            aggregate
                .peak_req
                .push(peak_throughput(&result.token_timestamps, 1.0));
        }
    }

    if let (Some(min_start), Some(max_first)) = (
        starts.iter().copied().reduce(f64::min),
        first_tokens.iter().copied().reduce(f64::max),
    ) {
        let duration = max_first - min_start;
        if duration > 0.0 {
            aggregate
                .pp_batch
                .push(prompt_tokens_total as f64 / duration);
        }
    }
    if let (Some(min_first), Some(max_last)) = (
        first_tokens.iter().copied().reduce(f64::min),
        last_tokens.iter().copied().reduce(f64::max),
    ) {
        let observed = results
            .iter()
            .map(|result| count_after_first(&result.token_timestamps))
            .sum::<usize>();
        let duration = max_last - min_first;
        if observed > 0 && duration > 0.0 {
            aggregate.tg_batch.push(observed as f64 / duration);
        }
    }
    if !all_timestamps.is_empty() {
        aggregate
            .peak_batch
            .push(peak_throughput(&all_timestamps, 1.0));
    }
}

fn count_after_first(timestamps: &[f64]) -> usize {
    timestamps.first().map_or(0, |first| {
        timestamps
            .iter()
            .filter(|timestamp| **timestamp > *first)
            .count()
    })
}

fn peak_throughput(timestamps: &[f64], window: f64) -> f64 {
    if timestamps.is_empty() {
        return 0.0;
    }
    let mut timestamps = timestamps.to_vec();
    timestamps.sort_by(f64::total_cmp);
    let duration = timestamps[timestamps.len() - 1] - timestamps[0];
    if duration > 0.0 && duration < window {
        return timestamps.len() as f64 / duration;
    }
    let mut start = 0usize;
    let mut maximum = 0usize;
    for end in 0..timestamps.len() {
        while start < end && timestamps[start] <= timestamps[end] - window {
            start += 1;
        }
        maximum = maximum.max(end - start + 1);
    }
    maximum as f64 / window
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(timestamps: Vec<f64>) -> RequestResult {
        RequestResult {
            start_ts: 0.0,
            end_ts: 1.4,
            first_response_ts: Some(1.0),
            first_token_ts: Some(1.0),
            prompt_tokens: 100,
            total_tokens: timestamps.len(),
            token_timestamps: timestamps,
        }
    }

    #[test]
    fn decode_rate_excludes_first_timestamp() {
        let run = aggregate(
            BenchmarkShape {
                prompt_tokens: 100,
                generated_tokens: 4,
                context_tokens: 0,
                concurrency: 1,
                expected_prompt_tokens: 100,
                is_context_phase: false,
            },
            &[vec![result(vec![1.0, 1.1, 1.2, 1.3])]],
            0.0,
        );
        assert!((run.tg_throughput.unwrap().mean - 10.0).abs() < 1e-8);
    }

    #[test]
    fn burst_has_no_decode_rate() {
        let run = aggregate(
            BenchmarkShape {
                prompt_tokens: 100,
                generated_tokens: 3,
                context_tokens: 0,
                concurrency: 1,
                expected_prompt_tokens: 100,
                is_context_phase: false,
            },
            &[vec![result(vec![1.0, 1.0, 1.0])]],
            0.0,
        );
        assert!(run.tg_throughput.is_none());
        assert_eq!(run.peak_throughput.unwrap().mean, 3.0);
    }
}
