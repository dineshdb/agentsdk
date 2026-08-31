use o3gen_openai::OpenAIApiClient;
use o3gen_openai::types::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    name: String,
    age: u8,
    city: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into());

    let client = OpenAIApiClient::new(base_url).with_api_key(api_key);

    let body = ChatCompletionRequest::builder()
        .model(model)
        .messages(vec![
            ChatCompletionRequestUserMessage::builder()
                .role(ChatCompletionRequestUserMessageRole::User)
                .content(ChatCompletionRequestUserMessageContent::String(
                    "John is 30 and lives in NYC. Sarah is 27 living in Kathmandu".into(),
                ))
                .build()
                .into(),
        ])
        .build()?;

    let person: Vec<Person> = client.json(body).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&person).expect("valid json")
    );

    Ok(())
}
