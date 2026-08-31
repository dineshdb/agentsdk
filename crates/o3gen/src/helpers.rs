use heck::ToPascalCase;
use proc_macro2::Span;
use syn::Ident;

#[must_use]
pub fn to_ident(name: &str) -> Ident {
    let mut sanitized = String::new();
    if name.is_empty() {
        return Ident::new("v_empty", Span::call_site());
    }
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        sanitized.push('v');
    }
    for c in name.chars() {
        if c.is_alphanumeric() {
            sanitized.push(c);
        } else {
            sanitized.push('_');
        }
    }
    if is_reserved(&sanitized) {
        if matches!(sanitized.as_str(), "self" | "Self" | "super" | "crate") {
            Ident::new(&format!("{sanitized}_"), Span::call_site())
        } else {
            Ident::new_raw(&sanitized, Span::call_site())
        }
    } else {
        Ident::new(&sanitized, Span::call_site())
    }
}

fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "abstract"
            | "as"
            | "async"
            | "await"
            | "become"
            | "box"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "do"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "macro"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "override"
            | "priv"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "typeof"
            | "unsafe"
            | "unsized"
            | "use"
            | "virtual"
            | "where"
            | "while"
            | "yield"
            | "try"
    )
}

#[must_use]
pub fn to_pascal_case(s: &str) -> String {
    let mut result = s.to_pascal_case();
    if result.is_empty() {
        return "Empty".to_string();
    }
    // Ensure first character is uppercase for valid Rust type names
    if let Some(first) = result.chars().next()
        && first.is_lowercase()
    {
        result = format!("{}{}", first.to_uppercase(), &result[1..]);
    }
    result
}
