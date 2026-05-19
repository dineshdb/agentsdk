use agentsdk::{Agent, AgentOptions, FileHistory, HistoryStore, Message, OpenAI, messages};
use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;
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

    // Setup conversation persistence
    let store = Arc::new(FileHistory::new(".agentsdk/history")?);
    let session_id = "git-summary-session";

    let mut handler = handler::GitHandler::new();

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

        // Add user message to history before running the agent
        store.push(session_id, messages::user(input)).await?;

        let history_store: Arc<dyn HistoryStore> = store.clone();
        let agent = Agent::builder()
            .client(client.clone())
            .options(
                AgentOptions::builder()
                    .history_store(history_store)
                    .session_id(session_id.to_string())
                    .with_tool(&tools::diff())
                    .with_tool(&tools::status())
                    .with_tool(&tools::log())
                    .build()?,
            )
            .build()?;

        agent.run(&mut handler).await?;

        // Print last message from store
        let history = store.load(session_id).await?;
        if let Some(content) = history.last().and_then(|m| {
            let Message::AssistantMessage(assistant) = m else {
                return None;
            };
            assistant.content.as_ref()
        }) {
            println!("\nAssistant: {content}");
        }
    }

    println!(
        "\nMetrics: Total API Errors: {}, Rate Limits: {}",
        handler.total_errors, handler.rate_limit_errors
    );

    Ok(())
}
