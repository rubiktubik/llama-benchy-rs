use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_stream::stream;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{Response, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use rust_benchy::{
    BenchmarkConfig,
    config::{LatencyMode, OutputFormat},
};
use serde_json::{Value, json};

#[derive(Clone)]
struct AppState {
    requests: Arc<std::sync::atomic::AtomicUsize>,
}

async fn models() -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": "test-model"}]}))
}

async fn completions(State(state): State<AppState>, Json(body): Json<Value>) -> impl IntoResponse {
    state
        .requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let messages = body["messages"].as_array().expect("messages");
    let prompt_chars = messages
        .iter()
        .filter_map(|message| message["content"].as_str())
        .map(str::chars)
        .map(Iterator::count)
        .sum::<usize>();
    let prompt_tokens = prompt_chars.div_ceil(4) + 4;
    if !stream_requested {
        return Json(json!({
            "choices": [{"message": {"role": "assistant", "content": "Paris"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1}
        }))
        .into_response();
    }

    let count = body["max_tokens"].as_u64().unwrap_or(1) as usize;
    let events = stream! {
        for token in 0..count {
            let event = json!({
                "choices": [{"delta": {"content": "x"}, "token_ids": [token]}]
            });
            yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(format!("data: {event}\n\n")));
            tokio::time::sleep(Duration::from_millis(3)).await;
        }
        let usage = json!({
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count}
        });
        yield Ok(Bytes::from(format!("data: {usage}\n\ndata: [DONE]\n\n")));
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(events))
        .unwrap()
        .into_response()
}

#[tokio::test]
async fn complete_suite_writes_json_report() {
    let state = AppState {
        requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completions))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let directory = tempfile::tempdir().unwrap();
    let corpus = directory.path().join("corpus.txt");
    tokio::fs::write(
        &corpus,
        "Sherlock Holmes considered the evidence. ".repeat(100),
    )
    .await
    .unwrap();
    let report_path = directory.path().join("report.json");
    let config = BenchmarkConfig {
        base_url: format!("http://{address}/v1"),
        api_key: "EMPTY".into(),
        model: "test-model".into(),
        served_model_name: "test-model".into(),
        pp_counts: vec![16],
        tg_counts: vec![4],
        depths: vec![0],
        concurrency_levels: vec![2],
        num_runs: 1,
        warmup_runs: 0,
        exact_tg: false,
        no_cache: false,
        enable_prefix_caching: false,
        latency_mode: LatencyMode::None,
        no_warmup: true,
        adapt_prompt: true,
        skip_coherence: true,
        prompt_file: Some(corpus),
        book_url: String::new(),
        post_run_cmd: None,
        extra_body: BTreeMap::new(),
        exit_on_first_fail: false,
        no_results_on_fail: false,
        save_result: Some(report_path.clone()),
        output_format: OutputFormat::Json,
        detailed: false,
        timeout: Duration::from_secs(5),
    };

    rust_benchy::run(config).await.unwrap();
    let report: Value =
        serde_json::from_slice(&tokio::fs::read(report_path).await.unwrap()).unwrap();
    assert_eq!(report["benchmarks"][0]["concurrency"], 2);
    assert!(
        report["benchmarks"][0]["tg_throughput"]["mean"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert_eq!(state.requests.load(std::sync::atomic::Ordering::Relaxed), 2);
}
