//! Logic-focused test of the o3gen-openai vision/image APIs.
//!
//! Exercises every vision code path the client supports: single-image,
//! multi-image content arrays, the `detail` field, SSE streaming parse, and
//! structured JSON extraction. Runs sequentially (no concurrency) so it verifies
//! client *logic*, not server bandwidth.
//!
//! Env: OPENAI_BASE_URL (default http://localhost:8181),
//!      OPENAI_MODEL (default LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0),
//!      VISION_IMG_DIR (default /tmp/vision_test)

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use o3gen_openai::types::*;
use o3gen_openai::{OpenAIApi, OpenAIApiClient};
use schemars::JsonSchema;
use serde::Deserialize;

struct Img {
    data_url: String,
}

fn load(dir: &str) -> Vec<Img> {
    let mut v = Vec::new();
    for f in ["red_circle.png", "blue_square.png", "green_triangle.png"] {
        let bytes = std::fs::read(format!("{dir}/{f}")).unwrap();
        v.push(Img {
            data_url: format!("data:image/png;base64,{}", B64.encode(bytes)),
        });
    }
    v
}

fn text_part(s: &str) -> ChatCompletionRequestMessageContentPart {
    ChatCompletionRequestMessageContentPart::Text(
        ChatCompletionRequestMessageContentPartText::builder()
            .r#type(ChatCompletionRequestMessageContentPartTextType::Text)
            .text(s.to_string())
            .build(),
    )
}

fn img_part(data_url: &str, detail: Option<Detail>) -> ChatCompletionRequestMessageContentPart {
    let image_url = match detail {
        Some(d) => ImageUrl::builder()
            .url(data_url.to_string())
            .detail(d)
            .build(),
        None => ImageUrl::builder().url(data_url.to_string()).build(),
    };
    ChatCompletionRequestMessageContentPart::Image(
        ChatCompletionRequestMessageContentPartImage::builder()
            .r#type(ChatCompletionRequestMessageContentPartImageType::ImageUrl)
            .image_url(image_url)
            .build(),
    )
}

fn user_msg(parts: Vec<ChatCompletionRequestMessageContentPart>) -> ChatCompletionRequestMessage {
    ChatCompletionRequestUserMessage::builder()
        .role(ChatCompletionRequestUserMessageRole::User)
        .content(ChatCompletionRequestUserMessageContent::Array(parts))
        .build()
        .into()
}

fn req(model: &str, parts: Vec<ChatCompletionRequestMessageContentPart>) -> ChatCompletionRequest {
    ChatCompletionRequest::builder()
        .model(model.to_string())
        .messages(vec![user_msg(parts)])
        .build()
        .expect("valid request")
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Scene {
    shape: String,
    color: String,
}

fn main_check(name: &str, ok: bool, detail: &str) {
    println!(
        "  [{:<4}] {:<32} {detail}",
        if ok { "PASS" } else { "FAIL" },
        name
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:8181".into());
    let model =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0".into());
    let dir = std::env::var("VISION_IMG_DIR").unwrap_or_else(|_| "/tmp/vision_test".into());
    let imgs = load(&dir);
    let client = OpenAIApiClient::new(base_url);
    println!("model={model}  images={}", imgs.len());

    // 1) Single-image vision correctness (color + shape).
    print!("1) single-image vision ... ");
    let mut color_ok = true;
    let mut shape_ok = true;
    for (img, (exp_c, exp_s)) in
        imgs.iter()
            .zip([("red", "circle"), ("blue", "square"), ("green", "triangle")])
    {
        let body = req(
            &model,
            vec![
                text_part("What color is the shape? Reply with only the color word."),
                img_part(&img.data_url, None),
            ],
        );
        let ans = OpenAIApi::create_chat_completion(&client, body)
            .await?
            .choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default()
            .to_lowercase();
        color_ok &= ans.contains(exp_c);

        let body = req(
            &model,
            vec![
                text_part("What shape is shown? Reply with only the shape word."),
                img_part(&img.data_url, None),
            ],
        );
        let ans = OpenAIApi::create_chat_completion(&client, body)
            .await?
            .choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default()
            .to_lowercase();
        shape_ok &= ans.contains(exp_s);
    }
    main_check("color recognized", color_ok, "");
    main_check("shape recognized", shape_ok, "");

    // 2) Multi-image content array (two images in one message).
    print!("2) multi-image array ... ");
    let body = req(
        &model,
        vec![
            text_part(
                "I show two images. The first is a red shape, the second a blue shape. \
                 Which image (first or second) is red? Reply with one word: first or second.",
            ),
            img_part(&imgs[0].data_url, None),
            img_part(&imgs[1].data_url, None),
        ],
    );
    let ans = OpenAIApi::create_chat_completion(&client, body)
        .await?
        .choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default()
        .to_lowercase();
    main_check(
        "multi-image (2 parts)",
        ans.contains("first"),
        &format!("(got '{ans}')"),
    );

    // 3) detail field serialization (low + high). Just verify the request is accepted.
    print!("3) detail field (low/high) ... ");
    let mut detail_ok = true;
    for d in [Detail::Low, Detail::High] {
        let body = req(
            &model,
            vec![
                text_part("What color is this shape? One word."),
                img_part(&imgs[0].data_url, Some(d)),
            ],
        );
        let r = OpenAIApi::create_chat_completion(&client, body).await;
        detail_ok &= r.is_ok();
    }
    main_check("detail low+high accepted", detail_ok, "");

    // 4) Streaming vision (SSE parse logic).
    print!("4) streaming vision ... ");
    let body = req(
        &model,
        vec![
            text_part("Describe this image in a few words."),
            img_part(&imgs[0].data_url, None),
        ],
    );
    let mut stream = client.stream_chat(body).await?;
    use futures_util::StreamExt;
    let mut collected = String::new();
    let mut chunks = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        chunks += 1;
        for ch in &chunk.choices {
            if let Some(c) = &ch.delta.content {
                collected.push_str(c);
            }
        }
    }
    main_check(
        "stream parsed + non-empty",
        !collected.is_empty() && chunks > 0,
        &format!("({chunks} chunks, '{collected:.30}')"),
    );

    // 5) Structured JSON extraction from an image.
    print!("5) structured JSON extraction ... ");
    let body = req(
        &model,
        vec![
            text_part("Describe the image. Provide the shape and its color."),
            img_part(&imgs[0].data_url, None),
        ],
    );
    match client.json::<Scene>(body).await {
        Ok(s) => main_check(
            "json parsed",
            true,
            &format!("(shape={:?}, color={:?})", s.shape, s.color),
        ),
        Err(e) => main_check("json parsed", false, &format!("(err: {e})")),
    }

    println!("done.");
    Ok(())
}
