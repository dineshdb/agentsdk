//! Stress test for `o3gen-openai` against a local llama-server (llama-swap).
//!
//! Exercises the vision/image APIs:
//!   1. Vision correctness — recognizable shapes/colors, verified against ground truth.
//!   2. Concurrent vision load — N simultaneous image requests, latency/throughput stats.
//!   3. Streaming vision — SSE streaming of image prompts under concurrency.
//!   4. Structured JSON extraction from an image (best-effort).
//!
//! Env:
//!   OPENAI_BASE_URL   default http://localhost:8181
//!   OPENAI_MODEL      default lfm-vl-3b
//!   VISION_IMG_DIR    default /tmp/vision_test
//!   STRESS_CONCURRENCY default 16
//!   STRESS_TOTAL      default 128

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use futures_util::future::join_all;
use o3gen_openai::types::*;
use o3gen_openai::{OpenAIApi, OpenAIApiClient};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Semaphore;

struct Image {
    name: String,
    data_url: String,
    expected_color: String,
    expected_shape: String,
}

fn load_images(dir: &str) -> Vec<Image> {
    let mut imgs = Vec::new();
    let expected = [
        ("red_circle.png", "red", "circle"),
        ("blue_square.png", "blue", "square"),
        ("green_triangle.png", "green", "triangle"),
        ("yellow_star.png", "yellow", "star"),
    ];
    for (fname, color, shape) in expected {
        let path = Path::new(dir).join(fname);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let data_url = format!("data:image/png;base64,{}", B64.encode(&bytes));
        imgs.push(Image {
            name: fname.to_string(),
            data_url,
            expected_color: color.to_string(),
            expected_shape: shape.to_string(),
        });
    }
    imgs
}

/// Build a chat request whose user message is a text prompt + optional images.
fn vision_body(model: &str, prompt: &str, images: &[&Image]) -> ChatCompletionRequest {
    let mut parts: Vec<ChatCompletionRequestMessageContentPart> =
        vec![ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText::builder()
                .r#type(ChatCompletionRequestMessageContentPartTextType::Text)
                .text(prompt.to_string())
                .build(),
        )];
    for img in images {
        parts.push(ChatCompletionRequestMessageContentPart::Image(
            ChatCompletionRequestMessageContentPartImage::builder()
                .r#type(ChatCompletionRequestMessageContentPartImageType::ImageUrl)
                .image_url(ImageUrl::builder().url(img.data_url.clone()).build())
                .build(),
        ));
    }

    let user = ChatCompletionRequestUserMessage::builder()
        .role(ChatCompletionRequestUserMessageRole::User)
        .content(ChatCompletionRequestUserMessageContent::Array(parts))
        .build();

    ChatCompletionRequest::builder()
        .model(model.to_string())
        .messages(vec![user.into()])
        .build()
        .expect("valid request")
}

/// 1) Vision correctness: each image gets a color + shape question.
async fn vision_correctness(client: &OpenAIApiClient, model: &str, imgs: &[Image]) {
    println!("\n=== 1) Vision correctness ===");
    let mut color_pass = 0;
    let mut shape_pass = 0;
    for img in imgs {
        // color
        let body = vision_body(
            model,
            "What color is the main shape in this image? Reply with only the color word.",
            &[img],
        );
        let color_ans = match OpenAIApi::create_chat_completion(client, body).await {
            Ok(r) => r.choices[0]
                .message
                .content
                .clone()
                .unwrap_or_default()
                .to_lowercase(),
            Err(e) => format!("ERROR: {e}"),
        };
        let c_ok = color_ans.contains(&img.expected_color);
        if c_ok {
            color_pass += 1;
        }

        // shape
        let body = vision_body(
            model,
            "What shape is shown in this image? Reply with only the shape word (circle, square, triangle, or star).",
            &[img],
        );
        let shape_ans = match OpenAIApi::create_chat_completion(client, body).await {
            Ok(r) => r.choices[0]
                .message
                .content
                .clone()
                .unwrap_or_default()
                .to_lowercase(),
            Err(e) => format!("ERROR: {e}"),
        };
        let s_ok = shape_ans.contains(&img.expected_shape);
        if s_ok {
            shape_pass += 1;
        }

        println!(
            "  {:<20} color={:<8} (got '{}')  shape={:<8} (got '{}')",
            img.name,
            if c_ok { "PASS" } else { "FAIL" },
            color_ans.trim(),
            if s_ok { "PASS" } else { "FAIL" },
            shape_ans.trim(),
        );
    }
    println!(
        "  => color {color_pass}/{}  shape {shape_pass}/{}",
        imgs.len(),
        imgs.len()
    );
}

fn stats(latencies_ms: &[f64]) -> (f64, f64, f64, f64) {
    if latencies_ms.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut v = latencies_ms.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = v[0];
    let max = v[v.len() - 1];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let p95 = v[std::cmp::min((v.len() as f64 * 0.95).ceil() as usize - 1, v.len() - 1)];
    (min, max, mean, p95)
}

/// 2) Concurrent vision load.
async fn stress_concurrent(
    client: Arc<OpenAIApiClient>,
    model: &str,
    imgs: Arc<Vec<Image>>,
    concurrency: usize,
    total: usize,
) {
    println!("\n=== 2) Concurrent vision load (concurrency={concurrency}, total={total}) ===");
    let sem = Arc::new(Semaphore::new(concurrency));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(total);
    let mut latencies: Vec<f64> = Vec::with_capacity(total);
    let mut errors = 0usize;

    for i in 0..total {
        let client = client.clone();
        let imgs = imgs.clone();
        let model = model.to_string();
        let permit = sem.clone().acquire_owned().await.unwrap();
        let h = tokio::spawn(async move {
            let img = &imgs[i % imgs.len()];
            let body = vision_body(
                &model,
                "In one short phrase, what do you see in this image?",
                &[img],
            );
            let t0 = Instant::now();
            let res = OpenAIApi::create_chat_completion(&*client, body).await;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            drop(permit);
            (res.is_ok(), ms)
        });
        handles.push(h);
    }

    for h in join_all(handles).await {
        match h {
            Ok((ok, ms)) => {
                if ok {
                    latencies.push(ms);
                } else {
                    errors += 1;
                }
            }
            Err(_) => errors += 1,
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let (min, max, mean, p95) = stats(&latencies);
    let ok = latencies.len();
    println!("  wall time        : {elapsed:.2}s");
    println!("  succeeded        : {ok}/{total}");
    println!("  errors           : {errors}");
    println!("  throughput       : {:.2} req/s", ok as f64 / elapsed);
    println!("  latency ms  min={min:.1}  mean={mean:.1}  p95={p95:.1}  max={max:.1}");
}

/// 3) Streaming vision under concurrency.
async fn stress_streaming(
    client: Arc<OpenAIApiClient>,
    model: &str,
    imgs: Arc<Vec<Image>>,
    concurrency: usize,
    total: usize,
) {
    println!("\n=== 3) Streaming vision (concurrency={concurrency}, total={total}) ===");
    let sem = Arc::new(Semaphore::new(concurrency));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(total);
    let mut ok = 0usize;
    let mut errors = 0usize;
    let mut total_tokens = 0usize;
    let mut first_err: Option<String> = None;

    for i in 0..total {
        let client = client.clone();
        let imgs = imgs.clone();
        let model = model.to_string();
        let permit = sem.clone().acquire_owned().await.unwrap();
        let h = tokio::spawn(async move {
            let img = &imgs[i % imgs.len()];
            let body = vision_body(&model, "Describe this image in a few words.", &[img]);
            let res = client.stream_chat(body).await;
            let mut stream = match res {
                Ok(s) => s,
                Err(e) => return (false, 0, Some(format!("{e}"))),
            };
            use futures_util::StreamExt;
            let mut toks = 0usize;
            let mut chunks = 0usize;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(c) => {
                        chunks += 1;
                        for ch in &c.choices {
                            if ch.delta.content.is_some() {
                                toks += 1;
                            }
                        }
                    }
                    Err(e) => return (false, 0, Some(format!("{e}"))),
                }
            }
            drop(permit);
            (chunks > 0, toks, None)
        });
        handles.push(h);
    }

    for h in join_all(handles).await {
        match h {
            Ok((success, toks, err)) => {
                if success {
                    ok += 1;
                    total_tokens += toks;
                } else {
                    errors += 1;
                    if first_err.is_none() {
                        first_err = err;
                    }
                }
            }
            Err(_) => errors += 1,
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("  wall time        : {elapsed:.2}s");
    println!("  streams ok       : {ok}/{total}");
    println!("  errors           : {errors}");
    if let Some(e) = &first_err {
        println!("  first error      : {e}");
    }
    println!("  content chunks   : {total_tokens}");
    println!("  throughput       : {:.2} streams/s", ok as f64 / elapsed);
}

/// 4) Structured JSON extraction from an image (best effort).
#[derive(Debug, Deserialize, JsonSchema)]
struct Scene {
    shape: String,
    color: String,
    background: String,
}

async fn structured_json(client: &OpenAIApiClient, model: &str, imgs: &[Image]) {
    println!("\n=== 4) Structured JSON extraction (best effort) ===");
    let img = &imgs[0];
    let body = vision_body(
        model,
        "Describe the image. Provide the shape, its color, and the background color.",
        &[img],
    );
    match client.json::<Scene>(body).await {
        Ok(scene) => println!(
            "  parsed Scene {{ shape={:?}, color={:?}, background={:?} }}",
            scene.shape, scene.color, scene.background
        ),
        Err(e) => println!("  JSON extraction failed (expected for small VL model): {e}"),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:8181".into());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "lfm-vl-3b".into());
    let img_dir = std::env::var("VISION_IMG_DIR").unwrap_or_else(|_| "/tmp/vision_test".into());
    let concurrency: usize = std::env::var("STRESS_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let total: usize = std::env::var("STRESS_TOTAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    println!("base_url={base_url}  model={model}  img_dir={img_dir}");

    let client = Arc::new(OpenAIApiClient::new(base_url));
    let imgs = load_images(&img_dir);
    println!("loaded {} test images", imgs.len());

    // warm-up: ensure the model backend is loaded
    print!("warming up backend...");
    let _ = OpenAIApi::create_chat_completion(
        &*client,
        vision_body(&model, "Say hi in one word.", &[&imgs[0]]),
    )
    .await;
    println!(" done");

    vision_correctness(&client, &model, &imgs).await;
    let imgs = Arc::new(imgs);
    stress_concurrent(client.clone(), &model, imgs.clone(), concurrency, total).await;
    stress_streaming(client.clone(), &model, imgs.clone(), concurrency, total).await;
    structured_json(&client, &model, &imgs).await;

    println!("\n=== DONE ===");
    Ok(())
}
