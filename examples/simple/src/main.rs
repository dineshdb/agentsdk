use agentsdk::{Agent, AgentListener, AgentOptions, OpenAI, messages};
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
    let mut history = Vec::new();

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

        history.push(messages::user(input));

        let agent = Agent::builder()
            .client(client.clone())
            .options(
                AgentOptions::builder()
                    .messages(Arc::new(history.clone()))
                    .build()?,
            )
            .build()?;

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut handler = SimpleHandler;
        let new_history = agent.run(&mut handler).await?;

        history.clone_from(&*new_history);
        println!();
    }

    Ok(())
}
