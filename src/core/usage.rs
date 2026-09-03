//! Cumulative LLM token usage for a run.
//!
//! Providers that honor `stream_options.include_usage` emit one final
//! usage chunk per request; the agent loop folds each into this component
//! on the agent entity. Read it from
//! [`AgentRunOutput::world`](crate::core::agent::AgentRunOutput) after the
//! run, or mid-run through a plugin's [`PluginContext`](crate::core::plugin::PluginContext).

use o3gen_openai::types::CreateChatCompletionStreamResponseUsage;

/// Token usage accumulated across every LLM request of a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    /// LLM requests that reported usage.
    pub requests: u32,
    /// Prompt (input) tokens, including cached ones.
    pub prompt_tokens: i64,
    /// Completion (output) tokens.
    pub completion_tokens: i64,
    /// Total tokens as reported by the provider.
    pub total_tokens: i64,
    /// Prompt tokens served from the provider's prompt cache.
    pub cached_tokens: i64,
    /// Completion tokens spent on reasoning / chain of thought.
    pub reasoning_tokens: i64,
}

impl From<&CreateChatCompletionStreamResponseUsage> for Usage {
    fn from(u: &CreateChatCompletionStreamResponseUsage) -> Self {
        Self {
            requests: 1,
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            cached_tokens: u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .unwrap_or(0),
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.requests += rhs.requests;
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;
        self.cached_tokens += rhs.cached_tokens;
        self.reasoning_tokens += rhs.reasoning_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use o3gen_openai::types::{
        CompletionTokensDetails, CreateChatCompletionStreamResponseUsage, PromptTokensDetails,
    };

    fn usage(
        prompt: i64,
        completion: i64,
        total: i64,
        cached: Option<i64>,
        reasoning: Option<i64>,
    ) -> CreateChatCompletionStreamResponseUsage {
        let mut b = CreateChatCompletionStreamResponseUsage::builder();
        b.prompt_tokens(prompt)
            .completion_tokens(completion)
            .total_tokens(total);
        if let Some(c) = cached {
            b.prompt_tokens_details(PromptTokensDetails {
                cached_tokens: Some(c),
            });
        }
        if let Some(r) = reasoning {
            b.completion_tokens_details(CompletionTokensDetails {
                reasoning_tokens: Some(r),
            });
        }
        b.build()
    }

    #[test]
    fn usage_from_chunk_counts_cache_and_reasoning() {
        let u = Usage::from(&usage(100, 50, 150, Some(80), Some(20)));
        assert_eq!(u.requests, 1);
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(u.total_tokens, 150);
        assert_eq!(u.cached_tokens, 80);
        assert_eq!(u.reasoning_tokens, 20);
    }

    #[test]
    fn usage_without_details_defaults_to_zero() {
        let u = Usage::from(&usage(10, 5, 15, None, None));
        assert_eq!(u.cached_tokens, 0);
        assert_eq!(u.reasoning_tokens, 0);
    }

    #[test]
    fn usage_accumulates_across_requests() {
        let mut total = Usage::from(&usage(100, 50, 150, Some(80), None));
        total += Usage::from(&usage(200, 30, 230, Some(160), Some(5)));
        assert_eq!(total.requests, 2);
        assert_eq!(total.prompt_tokens, 300);
        assert_eq!(total.completion_tokens, 80);
        assert_eq!(total.total_tokens, 380);
        assert_eq!(total.cached_tokens, 240);
        assert_eq!(total.reasoning_tokens, 5);
    }
}
