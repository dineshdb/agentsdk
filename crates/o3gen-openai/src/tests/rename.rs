use crate::types::{ChatCompletionRequestModel, Upstream};

#[test]
fn test_rename_upstream_regression() {
    // 1. Verify Upstream itself exists (as an enum)
    let _model = Upstream::Gpt41106Preview;

    // 2. Verify CreateChatCompletionRequestModel has an Upstream variant
    // instead of the auto-generated Variant1.
    let enum_variant: ChatCompletionRequestModel = Upstream::Gpt41106Preview.into();

    match enum_variant {
        ChatCompletionRequestModel::Upstream(u) => assert_eq!(u, Upstream::Gpt41106Preview),
        _ => panic!("Expected Upstream variant"),
    }
}
