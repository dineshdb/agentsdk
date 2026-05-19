use agentsdk::{Agent, AgentListener, AgentOptions, HistoryStore, MemoryHistory, OpenAI, messages};
use async_trait::async_trait;
use std::io::{self, Write};
use std::sync::Arc;

struct SimpleHandler;

#[async_trait]
impl AgentListener for SimpleHandler {
    async fn on_text_delta(&mut self, text: &str) {
        print!("{text}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let client = OpenAI::new(agentsdk::ModelConfig::from_env()?);
    let store = Arc::new(MemoryHistory::new());
    let session_id = "simple-session";

    println!("Simple Chat Agent. Type 'exit' to quit.");

    loop {
        print!("\nUser: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" {
            break;
        }

        // Add user message to history
        store.push(session_id, messages::user(input)).await?;

        let history_store: Arc<dyn HistoryStore> = store.clone();
        let agent = Agent::builder()
            .client(client.clone())
            .options(
                AgentOptions::builder()
                    .history_store(history_store)
                    .session_id(session_id.to_string())
                    .build()?,
            )
            .build()?;

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut handler = SimpleHandler;
        agent.run(&mut handler).await?;
        println!();
    }

    Ok(())
}
