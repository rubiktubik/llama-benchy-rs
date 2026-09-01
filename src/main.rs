use clap::Parser;
use rust_benchy::{Args, BenchmarkConfig, Result};

#[tokio::main]
async fn main() {
    if let Err(error) = try_main().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn try_main() -> Result<()> {
    let args = Args::parse();
    let config = BenchmarkConfig::resolve(args).await?;
    rust_benchy::run(config).await
}
