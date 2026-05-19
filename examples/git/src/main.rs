use agentsdk::{Agent, Message, OpenAI, messages};
use std::error::Error;
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

    tracing::info!("Starting git-summary example");

    let config = agentsdk::ModelConfig::from_env()?;
    let client = OpenAI::new(config);

    let agent = Agent::builder()
        .client(client)
        .options(
            agentsdk::AgentOptions::builder()
                .messages(std::sync::Arc::new(vec![messages::user(
                    "Show me the current git status and the last 3 commit messages using the tools provided.",
                )]))
                .with_tool(&tools::diff())
                .with_tool(&tools::status())
                .with_tool(&tools::log())
                .build()?,
        )
        .build()?;

    let mut handler = handler::GitHandler::new();
    let history = agent.run(&mut handler).await?;

    if let Some(content) = history.last().and_then(|m| {
        let Message::AssistantMessage(assistant) = m else {
            return None;
        };
        assistant.content.as_ref()
    }) {
        println!("\nFinal Response: {content}");
        println!(
            "\nMetrics: Total API Errors: {}, Rate Limits: {}",
            handler.total_errors, handler.rate_limit_errors
        );
    }

    Ok(())
}
