use chrono::{SecondsFormat, Utc};
use futures_util::future::join_all;

use crate::{
    BenchmarkConfig, Error, Result,
    client::{CONTEXT_LOAD_USER_MESSAGE, LlmClient, RequestResult},
    corpus::Corpus,
    metrics::{BenchmarkRun, BenchmarkShape, aggregate},
    prompt::{Calibration, PromptGenerator},
    report::BenchmarkReport,
};

pub async fn run(config: BenchmarkConfig) -> Result<()> {
    let client = LlmClient::new(&config)?;
    let corpus = Corpus::load(&config, client.http_client()).await?;
    let calibration = if config.no_warmup {
        eprintln!("Skipping token calibration; assuming four characters per token.");
        Calibration::default()
    } else {
        let short = corpus.calibration_sample(512);
        let long = corpus.calibration_sample(8192);
        client.calibrate(&short, &long).await?
    };

    if !config.skip_coherence {
        client.coherence_test().await?;
    }
    let latency = client
        .measure_latency(
            config.latency_mode,
            if config.no_warmup {
                0
            } else {
                config.warmup_runs
            },
        )
        .await?;
    eprintln!("Latency baseline: {:.2} ms", latency * 1000.0);

    let generator = PromptGenerator::new(corpus, calibration, config.adapt_prompt);
    let mut benchmarks = Vec::<BenchmarkRun>::new();
    let mut failed_requests = 0usize;

    for &depth in &config.depths {
        for &pp in &config.pp_counts {
            for &tg in &config.tg_counts {
                for &concurrency in &config.concurrency_levels {
                    eprintln!("Test pp={pp}, tg={tg}, depth={depth}, concurrency={concurrency}");
                    let mut standard_batches = Vec::with_capacity(config.num_runs);
                    let mut context_batches = Vec::with_capacity(config.num_runs);
                    let shape_warmups = if config.no_warmup {
                        0
                    } else {
                        config.warmup_runs
                    };
                    let total_runs = shape_warmups + config.num_runs;

                    for run_index in 0..total_runs {
                        let warmup = run_index < shape_warmups;
                        let label = if warmup {
                            format!("warmup {}/{}", run_index + 1, shape_warmups)
                        } else {
                            format!("run {}/{}", run_index - shape_warmups + 1, config.num_runs)
                        };
                        let prompts =
                            generator.generate_batch(concurrency, pp, depth, config.no_cache);

                        if config.enable_prefix_caching && depth > 0 {
                            eprintln!("  {label}: context load");
                            let context_prompts = prompts
                                .iter()
                                .map(|(context, _)| {
                                    (context.clone(), CONTEXT_LOAD_USER_MESSAGE.to_owned())
                                })
                                .collect();
                            let outcome =
                                execute_batch(&client, context_prompts, tg, config.no_cache).await;
                            failed_requests +=
                                handle_failures(&outcome.errors, config.exit_on_first_fail)?;
                            if !warmup {
                                context_batches.push(outcome.results);
                            }

                            eprintln!("  {label}: inference");
                            let outcome =
                                execute_batch(&client, prompts, tg, config.no_cache).await;
                            failed_requests +=
                                handle_failures(&outcome.errors, config.exit_on_first_fail)?;
                            if !warmup {
                                standard_batches.push(outcome.results);
                            }
                        } else {
                            eprintln!("  {label}");
                            let outcome =
                                execute_batch(&client, prompts, tg, config.no_cache).await;
                            failed_requests +=
                                handle_failures(&outcome.errors, config.exit_on_first_fail)?;
                            if !warmup {
                                standard_batches.push(outcome.results);
                            }
                        }

                        if let Some(command) = &config.post_run_cmd {
                            run_post_command(command).await?;
                        }
                    }

                    if config.enable_prefix_caching && depth > 0 {
                        benchmarks.push(aggregate(
                            BenchmarkShape {
                                prompt_tokens: pp,
                                generated_tokens: tg,
                                context_tokens: depth,
                                concurrency,
                                expected_prompt_tokens: depth,
                                is_context_phase: true,
                            },
                            &context_batches,
                            latency,
                        ));
                        benchmarks.push(aggregate(
                            BenchmarkShape {
                                prompt_tokens: pp,
                                generated_tokens: tg,
                                context_tokens: depth,
                                concurrency,
                                expected_prompt_tokens: pp,
                                is_context_phase: false,
                            },
                            &standard_batches,
                            latency,
                        ));
                    } else {
                        benchmarks.push(aggregate(
                            BenchmarkShape {
                                prompt_tokens: pp,
                                generated_tokens: tg,
                                context_tokens: depth,
                                concurrency,
                                expected_prompt_tokens: pp + depth,
                                is_context_phase: false,
                            },
                            &standard_batches,
                            latency,
                        ));
                    }
                }
            }
        }
    }

    if config.no_results_on_fail && failed_requests > 0 {
        return Err(Error::Benchmark(format!(
            "{failed_requests} request(s) failed; report suppressed by --no-results-on-fail"
        )));
    }
    let report = BenchmarkReport {
        version: env!("CARGO_PKG_VERSION").into(),
        timestamp: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        latency_mode: config.latency_mode.to_string(),
        latency_ms: latency * 1000.0,
        model: config.model.clone(),
        prefix_caching_enabled: config.enable_prefix_caching,
        max_concurrency: config.concurrency_levels.iter().copied().max().unwrap_or(1),
        failed_requests,
        benchmarks,
    };
    report
        .write(config.save_result.as_deref(), config.output_format)
        .await?;
    if failed_requests > 0 {
        return Err(Error::Benchmark(format!(
            "{failed_requests} request(s) failed"
        )));
    }
    Ok(())
}

struct BatchOutcome {
    results: Vec<RequestResult>,
    errors: Vec<Error>,
}

async fn execute_batch(
    client: &LlmClient,
    prompts: Vec<(String, String)>,
    max_tokens: usize,
    no_cache: bool,
) -> BatchOutcome {
    let futures = prompts.into_iter().map(|(context, prompt)| {
        let client = client.clone();
        async move {
            client
                .run_generation(&context, &prompt, max_tokens, no_cache)
                .await
        }
    });
    let mut results = Vec::new();
    let mut errors = Vec::new();
    for result in join_all(futures).await {
        match result {
            Ok(result) => results.push(result),
            Err(error) => errors.push(error),
        }
    }
    BatchOutcome { results, errors }
}

fn handle_failures(errors: &[Error], exit_on_first: bool) -> Result<usize> {
    for error in errors {
        eprintln!("  request failed: {error}");
    }
    if exit_on_first && let Some(error) = errors.first() {
        return Err(Error::Benchmark(error.to_string()));
    }
    Ok(errors.len())
}

async fn run_post_command(command: &str) -> Result<()> {
    let status = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .status()
        .await
        .map_err(|source| Error::StartPostRun {
            command: command.into(),
            source,
        })?;
    if !status.success() {
        return Err(Error::PostRun {
            command: command.into(),
            status: status.to_string(),
        });
    }
    Ok(())
}
