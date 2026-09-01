use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{BenchmarkConfig, Error, Result, config::LatencyMode, prompt::Calibration};

pub const CONTEXT_LOAD_USER_MESSAGE: &str = ".";

#[derive(Debug, Clone, Serialize)]
struct Message<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Clone)]
pub struct RequestResult {
    pub start_ts: f64,
    pub end_ts: f64,
    pub first_token_ts: Option<f64>,
    pub first_response_ts: Option<f64>,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub token_timestamps: Vec<f64>,
}

#[derive(Debug, Default)]
struct ParsedStream {
    prompt_tokens: usize,
    usage_completion_tokens: Option<usize>,
    first_response_ts: Option<f64>,
    first_token_ts: Option<f64>,
    content_chunks: Vec<ContentChunk>,
}

#[derive(Debug)]
struct ContentChunk {
    timestamp: f64,
    token_ids: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
struct NonStreamingResponse {
    #[serde(default)]
    choices: Vec<NonStreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct NonStreamingChoice {
    #[serde(default)]
    message: AssistantMessage,
}

#[derive(Debug, Default, Deserialize)]
struct AssistantMessage {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<usize>,
}

#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base_url: Arc<str>,
    model: Arc<str>,
    headers: HeaderMap,
    extra_body: Arc<BTreeMap<String, Value>>,
    exact_tg: bool,
    origin: Instant,
}

impl LlmClient {
    pub fn new(config: &BenchmarkConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if !config.api_key.is_empty() && config.api_key != "EMPTY" {
            let auth = HeaderValue::from_str(&format!("Bearer {}", config.api_key))
                .map_err(|_| Error::Config("API key contains invalid header characters".into()))?;
            headers.insert(AUTHORIZATION, auth);
        }
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .pool_idle_timeout(Duration::from_secs(600))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .map_err(|source| Error::Request {
                url: config.base_url.clone(),
                source,
            })?;
        Ok(Self {
            http,
            base_url: config.base_url.clone().into(),
            model: config.served_model_name.clone().into(),
            headers,
            extra_body: Arc::new(config.extra_body.clone()),
            exact_tg: config.exact_tg,
            origin: Instant::now(),
        })
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn timestamp(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }

    fn generation_payload(
        &self,
        messages: &[Message<'_>],
        max_tokens: usize,
        no_cache: bool,
    ) -> Value {
        let mut object = Map::new();
        object.insert("model".into(), json!(&*self.model));
        object.insert(
            "messages".into(),
            serde_json::to_value(messages).expect("messages serialize"),
        );
        object.insert("max_tokens".into(), json!(max_tokens));
        object.insert("stream".into(), json!(true));
        object.insert("return_token_ids".into(), json!(true));
        object.insert("stream_options".into(), json!({"include_usage": true}));
        if no_cache {
            object.insert("cache_prompt".into(), json!(false));
        }
        object.extend(
            self.extra_body
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        if self.exact_tg {
            object.insert("max_tokens".into(), json!(max_tokens));
            object.insert("min_tokens".into(), json!(max_tokens));
            object.insert("ignore_eos".into(), json!(true));
        }
        Value::Object(object)
    }

    pub async fn run_generation(
        &self,
        context: &str,
        prompt: &str,
        max_tokens: usize,
        no_cache: bool,
    ) -> Result<RequestResult> {
        let prompt = if prompt.trim().is_empty() {
            CONTEXT_LOAD_USER_MESSAGE
        } else {
            prompt
        };
        let mut messages = Vec::with_capacity(2);
        if !context.is_empty() {
            messages.push(Message {
                role: "system",
                content: context,
            });
        }
        messages.push(Message {
            role: "user",
            content: prompt,
        });
        let payload = self.generation_payload(&messages, max_tokens, no_cache);
        let url = self.endpoint("chat/completions");
        let start_ts = self.timestamp();
        let response = self
            .http
            .post(&url)
            .headers(self.headers.clone())
            .json(&payload)
            .send()
            .await
            .map_err(|source| Error::Request {
                url: url.clone(),
                source,
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read body: {error}>"));
            return Err(Error::HttpStatus { url, status, body });
        }

        let mut parsed = ParsedStream::default();
        let mut buffer = Vec::<u8>::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|source| Error::Request {
                url: url.clone(),
                source,
            })?;
            // All SSE events in one received byte chunk have the same observable
            // arrival time. Timing each parsed line separately would invent tiny
            // gaps and produce absurd generation-throughput values.
            let chunk_time = self.timestamp();
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..newline])
                    .trim_end_matches('\r')
                    .to_owned();
                buffer.drain(..=newline);
                self.parse_sse_line(&line, chunk_time, &url, &mut parsed)?;
            }
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            if !line.trim().is_empty() {
                self.parse_sse_line(line.trim(), self.timestamp(), &url, &mut parsed)?;
            }
        }
        let end_ts = self.timestamp();
        let (total_tokens, token_timestamps) = finalize_tokens(&parsed);

        Ok(RequestResult {
            start_ts,
            end_ts,
            first_token_ts: parsed.first_token_ts,
            first_response_ts: parsed.first_response_ts,
            prompt_tokens: parsed.prompt_tokens,
            total_tokens,
            token_timestamps,
        })
    }

    fn parse_sse_line(
        &self,
        line: &str,
        now: f64,
        endpoint: &str,
        parsed: &mut ParsedStream,
    ) -> Result<()> {
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(());
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(());
        }
        let value: Value = serde_json::from_str(data).map_err(|source| Error::Json {
            context: format!("SSE event from {endpoint}"),
            source,
        })?;
        if let Some(usage) = value.get("usage").and_then(Value::as_object) {
            if let Some(tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                parsed.prompt_tokens = tokens as usize;
            }
            if let Some(tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
                parsed.usage_completion_tokens = Some(tokens as usize);
            }
        }
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|v| v.first())
        else {
            return Ok(());
        };
        parsed.first_response_ts.get_or_insert(now);
        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            return Ok(());
        };
        let text = ["content", "reasoning_content", "reasoning"]
            .into_iter()
            .find_map(|key| {
                delta
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
            });
        if text.is_some() {
            parsed.first_token_ts.get_or_insert(now);
            let token_ids = choice.get("token_ids").and_then(Value::as_array).cloned();
            parsed.content_chunks.push(ContentChunk {
                timestamp: now,
                token_ids,
            });
        }
        Ok(())
    }

    pub async fn calibrate(&self, short: &str, long: &str) -> Result<Calibration> {
        eprintln!("Calibrating prompt sizes using endpoint token counts ...");
        let user_short = self.prompt_token_count("", short).await?;
        let user_long = self.prompt_token_count("", long).await?;
        let context_short = self
            .prompt_token_count(short, CONTEXT_LOAD_USER_MESSAGE)
            .await?;
        let context_long = self
            .prompt_token_count(long, CONTEXT_LOAD_USER_MESSAGE)
            .await?;
        let calibration = Calibration::from_samples(
            short.chars().count(),
            long.chars().count(),
            user_short,
            user_long,
            context_short,
            context_long,
        );
        eprintln!(
            "Calibration: user {:.2} chars/token, context {:.2} chars/token",
            calibration.user_chars_per_token, calibration.context_chars_per_token,
        );
        Ok(calibration)
    }

    async fn prompt_token_count(&self, context: &str, prompt: &str) -> Result<usize> {
        let mut messages = Vec::with_capacity(2);
        if !context.is_empty() {
            messages.push(Message {
                role: "system",
                content: context,
            });
        }
        messages.push(Message {
            role: "user",
            content: prompt,
        });
        let payload = json!({
            "model": &*self.model,
            "messages": messages,
            "max_tokens": 1,
            "stream": false,
        });
        let response: NonStreamingResponse = self.post_json("chat/completions", &payload).await?;
        response.usage.and_then(|u| u.prompt_tokens).ok_or_else(|| Error::Protocol {
            endpoint: self.endpoint("chat/completions"),
            message: "calibration requires usage.prompt_tokens in non-streaming responses; use --no-warmup to assume four characters/token".into(),
        })
    }

    pub async fn coherence_test(&self) -> Result<()> {
        eprintln!("Running coherence check ...");
        let payload = json!({
            "model": &*self.model,
            "messages": [{"role": "user", "content": "What is the capital of France? Reply with one word."}],
            "max_tokens": 100,
            "stream": false,
        });
        let response: NonStreamingResponse = self.post_json("chat/completions", &payload).await?;
        let answer = response
            .choices
            .first()
            .map(|choice| {
                let message = &choice.message;
                format!(
                    "{}{}{}",
                    message.content.as_deref().unwrap_or_default(),
                    message.reasoning.as_deref().unwrap_or_default(),
                    message.reasoning_content.as_deref().unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        if !answer.to_lowercase().contains("paris") {
            return Err(Error::Benchmark(format!(
                "coherence check expected 'Paris', got {:?}",
                answer.chars().take(200).collect::<String>()
            )));
        }
        Ok(())
    }

    pub async fn measure_latency(&self, mode: LatencyMode, warmup_runs: usize) -> Result<f64> {
        match mode {
            LatencyMode::None => Ok(0.0),
            LatencyMode::Api => {
                eprintln!("Measuring API latency ...");
                let start = Instant::now();
                let url = self.endpoint("models");
                let response = self
                    .http
                    .get(&url)
                    .headers(self.headers.clone())
                    .send()
                    .await
                    .map_err(|source| Error::Request {
                        url: url.clone(),
                        source,
                    })?;
                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(Error::HttpStatus { url, status, body });
                }
                Ok(start.elapsed().as_secs_f64())
            }
            LatencyMode::Generation => {
                eprintln!("Measuring generation latency ...");
                let measured_runs = 3;
                let mut samples = Vec::with_capacity(measured_runs);
                for index in 0..(warmup_runs + measured_runs) {
                    let start = Instant::now();
                    let messages = [Message {
                        role: "user",
                        content: "hello",
                    }];
                    let payload = json!({
                        "model": &*self.model, "messages": messages,
                        "max_tokens": 1, "stream": true,
                    });
                    let url = self.endpoint("chat/completions");
                    let response = self
                        .http
                        .post(&url)
                        .headers(self.headers.clone())
                        .json(&payload)
                        .send()
                        .await
                        .map_err(|source| Error::Request {
                            url: url.clone(),
                            source,
                        })?;
                    let status = response.status();
                    if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        return Err(Error::HttpStatus { url, status, body });
                    }
                    let mut bytes = response.bytes_stream();
                    let first =
                        bytes
                            .next()
                            .await
                            .transpose()
                            .map_err(|source| Error::Request {
                                url: url.clone(),
                                source,
                            })?;
                    if first.is_none() {
                        return Err(Error::Protocol {
                            endpoint: url,
                            message: "empty streaming response".into(),
                        });
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    while let Some(next) = bytes.next().await {
                        next.map_err(|source| Error::Request {
                            url: url.clone(),
                            source,
                        })?;
                    }
                    if index >= warmup_runs {
                        samples.push(elapsed);
                    }
                }
                Ok(samples.iter().sum::<f64>() / samples.len() as f64)
            }
        }
    }

    async fn post_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        payload: &Value,
    ) -> Result<T> {
        let url = self.endpoint(path);
        let response = self
            .http
            .post(&url)
            .headers(self.headers.clone())
            .json(payload)
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
        serde_json::from_str(&body).map_err(|source| Error::Json {
            context: url,
            source,
        })
    }
}

fn finalize_tokens(parsed: &ParsedStream) -> (usize, Vec<f64>) {
    if parsed.content_chunks.is_empty() {
        return (parsed.usage_completion_tokens.unwrap_or(0), Vec::new());
    }
    if parsed
        .content_chunks
        .iter()
        .all(|chunk| chunk.token_ids.is_some())
    {
        let mut timestamps = Vec::new();
        for chunk in &parsed.content_chunks {
            let count = chunk.token_ids.as_ref().map_or(0, Vec::len);
            append_interpolated(
                &mut timestamps,
                chunk.timestamp,
                count,
                parsed.first_token_ts,
            );
        }
        return (timestamps.len(), timestamps);
    }
    let chunk_times = parsed
        .content_chunks
        .iter()
        .map(|chunk| chunk.timestamp)
        .collect::<Vec<_>>();
    if let Some(count) = parsed.usage_completion_tokens {
        return (count, interpolate_range(&chunk_times, count));
    }
    // The OpenAI stream protocol has no per-chunk token guarantee. Without usage or
    // token_ids this is explicitly a chunk-count estimate.
    (chunk_times.len(), chunk_times)
}

fn append_interpolated(
    timestamps: &mut Vec<f64>,
    chunk_time: f64,
    count: usize,
    first: Option<f64>,
) {
    if count == 0 {
        return;
    }
    let last = timestamps.last().copied().or(first).unwrap_or(chunk_time);
    let width = (chunk_time - last).max(0.0);
    timestamps.extend((1..=count).map(|i| last + width * i as f64 / count as f64));
}

fn interpolate_range(chunk_times: &[f64], count: usize) -> Vec<f64> {
    if count == 0 || chunk_times.is_empty() {
        return Vec::new();
    }
    if count == 1 || chunk_times.len() == 1 {
        return vec![chunk_times[0]; count];
    }
    let first = chunk_times[0];
    let last = chunk_times[chunk_times.len() - 1];
    if last <= first {
        return vec![first; count];
    }
    let step = (last - first) / (count - 1) as f64;
    (0..count).map(|i| first + step * i as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_interpolates_tokens_across_chunks() {
        let parsed = ParsedStream {
            usage_completion_tokens: Some(3),
            content_chunks: vec![
                ContentChunk {
                    timestamp: 1.0,
                    token_ids: None,
                },
                ContentChunk {
                    timestamp: 2.0,
                    token_ids: None,
                },
            ],
            ..Default::default()
        };
        let (count, timestamps) = finalize_tokens(&parsed);
        assert_eq!(count, 3);
        assert_eq!(timestamps, vec![1.0, 1.5, 2.0]);
    }

    #[test]
    fn token_ids_take_precedence_over_usage() {
        let parsed = ParsedStream {
            usage_completion_tokens: Some(99),
            first_token_ts: Some(1.0),
            content_chunks: vec![ContentChunk {
                timestamp: 2.0,
                token_ids: Some(vec![json!(1), json!(2), json!(3)]),
            }],
            ..Default::default()
        };
        let (count, timestamps) = finalize_tokens(&parsed);
        assert_eq!(count, 3);
        for (actual, expected) in timestamps.iter().zip([4.0 / 3.0, 5.0 / 3.0, 2.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
