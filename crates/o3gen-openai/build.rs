use o3gen::Generator;

fn main() {
    // Generate types from petstore OpenAPI
    Generator::builder("openai.json")
        .api_name("OpenAIApi")
        .deny_unknown_fields(false)
        .rename("CreateChatCompletionRequest", "ChatCompletionRequest")
        .rename("ChatCompletionRequestModelVariant1", "Upstream")
        .rename("CreateCompletionRequestModelVariant1", "ModelId")
        .rename("ChatCompletionMessageToolCallFunction", "ToolCallFunction")
        .rename("ChatCompletionMessageToolCall", "ToolCall")
        .write_to_out_dir("openai.rs")
        .expect("Failed to generate openai.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=openai.json");
}
