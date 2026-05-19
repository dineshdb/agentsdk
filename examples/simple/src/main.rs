use agentsdk::{Agent, AgentPlugin, MemoryHistoryPlugin, OpenAI, PluginContext, messages};
use async_trait::async_trait;
use std::io::{self, Write};

struct PrinterPlugin;

#[async_trait]
impl AgentPlugin for PrinterPlugin {
    fn name(&self) -> &'static str {
        "printer"
    }

    async fn on_text_delta(&mut self, _ctx: &PluginContext, text: &str) {
        print!("{text}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let client = OpenAI::new(agentsdk::ModelConfig::from_env()?);
    let history = MemoryHistoryPlugin::new();

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

        // Push the user message into the shared history plugin
        history.push(messages::user(input)).await;

        let mut agent = Agent::builder()
            .client(client.clone())
            .options(agentsdk::AgentOptions::builder().build()?)
            .plugin(history.clone()) // share history via Arc
            .plugin(PrinterPlugin) // stream text to stdout
            .build()?;

        print!("Assistant: ");
        io::stdout().flush()?;

        agent.run().await?;
        println!();
    }

    Ok(())
}
