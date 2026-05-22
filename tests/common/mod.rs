use agentsdk::OpenAI;

pub fn init_llm_test() -> Option<OpenAI> {
    dotenv::dotenv().ok();

    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Skipping LLM test: OPENAI_API_KEY not set");
        return None;
    }

    let config = agentsdk::ModelConfig::from_env().ok()?;
    Some(OpenAI::new(config))
}
