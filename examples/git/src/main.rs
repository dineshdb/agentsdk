use agentsdk::{Agent, AgentOptions, FileHistoryPlugin, Message, OpenAI, messages};
use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::Level;
use tracing_subscriber::EnvFilter;

mod config;
mod handler;
mod tools;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .with_target(false)
        .without_time()
        .init();

    tracing::info!("Starting interactive git-summary example");

    let config = agentsdk::ModelConfig::from_env()?;
    let client = OpenAI::new(config);

    // Shared history plugin (file-backed, persists across turns)
    let history = FileHistoryPlugin::new(".agentsdk/history")?;

    let metrics = Arc::new(handler::GitMetrics::default());

    println!("Interactive Git Agent. Type 'exit' or 'quit' to end.");

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            break;
        }

        // Push user message to history before running the agent
        history.push(messages::user(input)).await?;

        let mut agent = Agent::builder()
            .client(client.clone())
            .options(
                AgentOptions::builder()
                    .with_tool(&tools::diff())
                    .with_tool(&tools::status())
                    .with_tool(&tools::log())
                    .build()?,
            )
            .plugin(history.clone()) // share history via Arc
            .plugin(handler::GitHandler::new(metrics.clone()))
            .build()?;

        agent.run().await?;

        println!(
            "\nMetrics: Total API Errors: {}, Rate Limits: {}",
            metrics.total_errors.load(Ordering::Relaxed),
            metrics.rate_limit_errors.load(Ordering::Relaxed),
        );

        // Print the last assistant message from the plugin
        let msgs = history.load().await?;
        if let Some(content) = msgs.last().and_then(|m| {
            let Message::AssistantMessage(assistant) = m else {
                return None;
            };
            assistant.content.as_ref()
        }) {
            println!("\nAssistant: {content}");
        }
    }

    Ok(())
}
