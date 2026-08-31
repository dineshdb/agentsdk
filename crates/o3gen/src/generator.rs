use proc_macro2::TokenStream;
use quote::quote;
use std::fs;
use std::path::{Path, PathBuf};
use syn::{File, parse2};

use crate::client::{
    OperationDetails, ParameterDetails, generate_client_impl, generate_client_traits,
};
use crate::config::Config;
use crate::emit::EmitContext;
use crate::helpers::to_ident;
use crate::ir::{ApiIr, TypeIr};
use crate::transformer::Transformer;

#[derive(Debug)]
pub struct Generator {
    config: Config,
}

impl Generator {
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn builder(path: impl Into<String>) -> Self {
        Self {
            config: Config {
                path: path.into(),
                ..Config::default()
            },
        }
    }

    #[must_use]
    pub fn rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.config.rename.insert(from.into(), to.into());
        self
    }

    #[must_use]
    pub fn derive_extra(
        mut self,
        type_name: impl Into<String>,
        traits: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.config.derive_extra.insert(
            type_name.into(),
            traits.into_iter().map(Into::into).collect(),
        );
        self
    }

    #[must_use]
    pub fn api_name(mut self, name: impl Into<String>) -> Self {
        self.config.api_name = Some(name.into());
        self
    }

    #[must_use]
    pub fn deny_unknown_fields(mut self, deny: bool) -> Self {
        self.config.deny_unknown_fields = deny;
        self
    }

    /// Generates the Rust code for the API.
    ///
    /// # Errors
    /// Returns an error if the `OpenAPI` file cannot be read, parsed, or if code generation fails.
    pub fn generate(&mut self) -> Result<String, String> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let full_path = Path::new(&manifest_dir).join(&self.config.path);

        let spec_str = fs::read_to_string(&full_path).map_err(|e| {
            format!(
                "Failed to read OpenAPI file at {}: {e}",
                full_path.display()
            )
        })?;

        let openapi: openapiv3::OpenAPI = serde_json::from_str(&spec_str)
            .map_err(|e| format!("Failed to parse OpenAPI JSON: {e}"))?;

        let ir = Transformer::transform(&openapi, &self.config)?;

        let api_name = self
            .config
            .api_name
            .clone()
            .unwrap_or_else(|| crate::helpers::to_pascal_case(&openapi.info.title));

        self.emit(&ir, &api_name)
    }

    fn emit(&self, ir: &ApiIr, api_name: &str) -> Result<String, String> {
        let ctx = EmitContext {
            deny_unknown_fields: self.config.deny_unknown_fields,
        };

        let mut types_tokens = TokenStream::new();
        for (_, def) in &ir.types {
            types_tokens.extend(def.emit(ctx));
        }

        let mut output_tokens = TokenStream::new();
        output_tokens.extend(quote! {
            use thiserror::Error;

            #[derive(Debug, Error)]
            pub enum ApiError {
                #[error("Request error: {0}")]
                Reqwest(#[from] reqwest::Error),
                #[error("Status error: {status} - {body}")]
                Status { status: reqwest::StatusCode, body: String },
                #[error("Serialization error: {0}")]
                Serde(#[from] serde_json::Error),
                #[error("Validation error: {0}")]
                Validation(#[from] ValidationError),
                #[error("Builder error: {0}")]
                Builder(String),
            }

            #[derive(Debug, Error)]
            pub enum ValidationError {
                #[error("Field '{field}' is too short (min: {min}, max: {max})")]
                LengthTooShort { field: String, min: u64, max: u64 },
                #[error("Field '{field}' is too long (min: {min}, max: {max})")]
                LengthTooLong { field: String, min: u64, max: u64 },
                #[error("Field '{field}' is below minimum (min: {min}, max: {max})")]
                RangeTooSmall { field: String, min: f64, max: f64 },
                #[error("Field '{field}' is above maximum (min: {min}, max: {max})")]
                RangeTooLarge { field: String, min: f64, max: f64 },
                #[error("Field '{field}' is invalid: {message}")]
                Invalid { field: String, message: String },
            }

            pub type Result<T> = std::result::Result<T, ApiError>;

            #[allow(
                nonstandard_style,
                unused,
                dead_code,
                non_camel_case_types,
                unused_imports,
                unused_mut,
                unreachable_pub,
                unused_qualifications,
                unused_variables,
            )]
            pub mod types {
                use serde::{Serialize, Deserialize};
                use validator::Validate;
                use derive_builder::Builder;
                use super::{ApiError, ValidationError, Result};
                #types_tokens
            }

            pub use types::*;
        });

        output_tokens.extend(Self::emit_client_code(ir, api_name));

        let file = parse2::<File>(output_tokens)
            .map_err(|e| format!("Failed to parse generated tokens: {e}"))?;

        Ok(prettyplease::unparse(&file))
    }

    fn emit_client_code(ir: &ApiIr, api_name: &str) -> TokenStream {
        let mut operations_details = Vec::new();
        for op in &ir.operations {
            let response_type = op.responses.iter().find_map(|r| {
                if r.code.is_success() {
                    r.type_info.as_ref().map(TypeIr::to_type_string)
                } else {
                    None
                }
            });

            let parameters = op
                .parameters
                .iter()
                .map(|p| ParameterDetails {
                    name: p.name.clone(),
                    rust_type: p.type_info.to_type_string(),
                    description: p.description.clone(),
                })
                .collect();

            operations_details.push(OperationDetails {
                operation_id: op.operation_id.clone(),
                http_method: op.method.clone(),
                response_type,
                parameters,
                request_body_type: op.request_body.as_ref().map(TypeIr::to_type_string),
                path: op.path.clone(),
                description: op.description.clone(),
            });
        }
        operations_details.sort_by(|a, b| a.operation_id.cmp(&b.operation_id));

        if operations_details.is_empty() {
            return TokenStream::new();
        }

        let trait_ident = to_ident(api_name);
        let client_name = format!("{api_name}Client");
        let client_ident = to_ident(&client_name);

        let mut tokens = TokenStream::new();
        tokens.extend(generate_client_traits(&trait_ident, &operations_details));
        tokens.extend(generate_client_impl(
            &trait_ident,
            &client_ident,
            &operations_details,
            &ir.security_schemes,
        ));
        tokens
    }

    /// Writes the generated code to a file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be written or code generation fails.
    pub fn write_to_file(mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let code = self.generate()?;
        let path = path.as_ref();
        fs::write(path, code).map_err(|e| format!("Failed to write to {}: {e}", path.display()))?;
        println!("cargo:rerun-if-changed={}", self.config.path);
        Ok(())
    }

    /// Writes the generated code to the `OUT_DIR`.
    ///
    /// # Errors
    /// Returns an error if `OUT_DIR` is not set or writing fails.
    pub fn write_to_out_dir(self, filename: impl AsRef<Path>) -> Result<(), String> {
        let out_dir = std::env::var_os("OUT_DIR")
            .ok_or_else(|| "OUT_DIR environment variable is not set".to_string())?;
        let dest_path = PathBuf::from(out_dir).join(filename);
        self.write_to_file(dest_path)
    }
}
