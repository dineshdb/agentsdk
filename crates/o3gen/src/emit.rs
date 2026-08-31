use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::helpers::to_ident;
use crate::ir::{
    AliasIr, AnyOfIr, EnumIr, FieldIr, IrType, NewtypeIr, PrimitiveType, StructIr,
    TypeDefinitionIr, TypeIr, ValidationIr,
};

#[derive(Debug, Clone, Copy)]
pub struct EmitContext {
    pub deny_unknown_fields: bool,
}

#[must_use]
pub fn emit_doc(desc: Option<&str>) -> TokenStream {
    if let Some(d) = desc {
        let d = d
            .trim()
            .replace("```\n", "```text\n")
            .replace("```\r\n", "```text\n");
        quote! { #[doc = #d] }
    } else {
        TokenStream::new()
    }
}

#[must_use]
pub fn emit_derives(derives: &[String]) -> TokenStream {
    let paths: Vec<TokenStream> = derives
        .iter()
        .map(|d| {
            if d.contains("::") {
                let parts: Vec<_> = d
                    .split("::")
                    .map(|p| Ident::new(p, Span::call_site()))
                    .collect();
                quote! { #(#parts)::* }
            } else {
                let ident = Ident::new(d, Span::call_site());
                quote! { #ident }
            }
        })
        .collect();
    quote! { #[derive(#(#paths),*)] }
}

#[must_use]
pub fn emit_derives_for<T: IrType>(ir: &T) -> TokenStream {
    let mut derives = ir.derives().to_vec();
    derives.retain(|d| ir.should_derive(d));
    emit_derives(&derives)
}

#[must_use]
pub fn emit_validation(validation: &[ValidationIr], _type_info: &TypeIr) -> TokenStream {
    if validation.is_empty() {
        return TokenStream::new();
    }
    let mut parts = Vec::new();
    for v in validation {
        match v {
            ValidationIr::Length { min, max } => {
                let mut l_parts = Vec::new();
                if let Some(m) = min {
                    l_parts.push(quote! { min = #m });
                }
                if let Some(m) = max {
                    l_parts.push(quote! { max = #m });
                }
                parts.push(quote! { length(#(#l_parts),*) });
            }
            ValidationIr::FloatRange { min, max } => {
                let mut r_parts = Vec::new();
                if let Some(m) = min {
                    let lit = syn::LitFloat::new(&format!("{m:.1}f64"), Span::call_site());
                    r_parts.push(quote! { min = #lit });
                }
                if let Some(m) = max {
                    let lit = syn::LitFloat::new(&format!("{m:.1}f64"), Span::call_site());
                    r_parts.push(quote! { max = #lit });
                }
                parts.push(quote! { range(#(#r_parts),*) });
            }
            ValidationIr::IntRange { min, max } => {
                let mut r_parts = Vec::new();
                if let Some(m) = min {
                    let lit = syn::LitInt::new(&format!("{m}i64"), Span::call_site());
                    r_parts.push(quote! { min = #lit });
                }
                if let Some(m) = max {
                    let lit = syn::LitInt::new(&format!("{m}i64"), Span::call_site());
                    r_parts.push(quote! { max = #lit });
                }
                parts.push(quote! { range(#(#r_parts),*) });
            }
        }
    }
    if parts.is_empty() {
        TokenStream::new()
    } else {
        quote! { #[validate(#(#parts),*)] }
    }
}

// --- TypeIr methods ---

impl TypeIr {
    #[must_use]
    pub fn to_tokens(&self, required: bool) -> TokenStream {
        let inner = match self {
            TypeIr::Reference(r) | TypeIr::Enum(r) => {
                let ident = to_ident(r);
                quote! { #ident }
            }
            TypeIr::Primitive(p) => match p {
                PrimitiveType::String => quote! { String },
                PrimitiveType::Integer => quote! { i64 },
                PrimitiveType::Number => quote! { f64 },
                PrimitiveType::Boolean => quote! { bool },
                PrimitiveType::Date => quote! { chrono::NaiveDate },
                PrimitiveType::DateTime => quote! { chrono::DateTime<chrono::Utc> },
            },
            TypeIr::Array(inner) => {
                let inner_tokens = inner.to_tokens(true);
                quote! { Vec<#inner_tokens> }
            }
            TypeIr::Map(inner) => {
                let inner_tokens = inner.to_tokens(true);
                quote! { std::collections::HashMap<String, #inner_tokens> }
            }
            TypeIr::Value => quote! { serde_json::Value },
        };

        if required {
            inner
        } else {
            quote! { Option<#inner> }
        }
    }

    #[must_use]
    pub fn to_type_string(&self) -> String {
        match self {
            TypeIr::Reference(r) | TypeIr::Enum(r) => r.clone(),
            TypeIr::Primitive(p) => match p {
                PrimitiveType::String => "String".to_string(),
                PrimitiveType::Integer => "i64".to_string(),
                PrimitiveType::Number => "f64".to_string(),
                PrimitiveType::Boolean => "bool".to_string(),
                PrimitiveType::Date => "chrono::NaiveDate".to_string(),
                PrimitiveType::DateTime => "chrono::DateTime<chrono::Utc>".to_string(),
            },
            TypeIr::Array(inner) => format!("Vec<{}>", inner.to_type_string()),
            TypeIr::Map(inner) => format!(
                "std::collections::HashMap<String, {}>",
                inner.to_type_string()
            ),
            TypeIr::Value => "serde_json::Value".to_string(),
        }
    }
}

// --- Field emit ---

impl FieldIr {
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        let f_name = to_ident(&self.rust_name);
        let f_type = self.type_info.to_tokens(self.required);
        let f_doc_attr = emit_doc(self.description.as_deref());
        let serde_attr = self
            .serde_rename
            .as_ref()
            .map(|r| quote! { #[serde(rename = #r)] });
        let validate_attr = emit_validation(&self.validation, &self.type_info);
        let builder_attr = if self.required {
            TokenStream::new()
        } else {
            quote! { #[builder(default)] }
        };
        let skip_if_none = if self.required {
            quote! {}
        } else {
            quote! { #[serde(skip_serializing_if = "Option::is_none")] }
        };
        quote! {
            #f_doc_attr
            #serde_attr
            #skip_if_none
            #validate_attr
            #builder_attr
            pub #f_name: #f_type,
        }
    }
}

impl TypeDefinitionIr {
    #[must_use]
    pub fn emit(&self, ctx: EmitContext) -> TokenStream {
        match self {
            Self::Struct(s) => s.emit(ctx),
            Self::Enum(e) => e.emit(),
            Self::Alias(a) => a.emit(),
            Self::Newtype(n) => n.emit(),
            Self::AnyOf(a) => a.emit(),
        }
    }
}

impl StructIr {
    #[must_use]
    pub fn emit(&self, ctx: EmitContext) -> TokenStream {
        let name = to_ident(self.name.as_str());
        let builder_name = to_ident(&format!("{}Builder", self.name.as_str()));
        let derives = emit_derives_for(self);
        let doc_attr = emit_doc(self.description.as_deref());
        let fields: Vec<TokenStream> = self.fields.iter().map(FieldIr::emit).collect();

        let has_additional = self.additional_properties_type.is_some();
        let deny_unknown = if !has_additional && ctx.deny_unknown_fields {
            quote! { #[serde(deny_unknown_fields)] }
        } else {
            quote! {}
        };

        let additional_properties_field = if let Some(ap_type) = &self.additional_properties_type {
            let ap_tokens = ap_type.to_tokens(true);
            quote! {
                #[serde(flatten)]
                #[builder(default)]
                pub additional_properties: #ap_tokens,
            }
        } else {
            quote! {}
        };

        let build_method = if self.has_validation() {
            quote! {
                pub fn build(&self) -> Result<#name> {
                    let obj = self.build_inner().map_err(|e| ApiError::Builder(e.to_string()))?;
                    obj.validate().map_err(|e| {
                        if let Some((field, field_errors)) = e.field_errors().into_iter().next() {
                            if let Some(err) = field_errors.iter().next() {
                                match err.code.as_ref() {
                                    "length" => {
                                        let min = err.params.get("min").and_then(|v| v.as_u64()).unwrap_or(0);
                                        let max = err.params.get("max").and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
                                        return ApiError::Validation(ValidationError::LengthTooShort {
                                            field: field.to_string(),
                                            min,
                                            max,
                                        });
                                    }
                                    "range" => {
                                        let min = err.params.get("min").and_then(|v| v.as_f64()).unwrap_or(f64::MIN);
                                        let max = err.params.get("max").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
                                        return ApiError::Validation(ValidationError::RangeTooSmall {
                                            field: field.to_string(),
                                            min,
                                            max,
                                        });
                                    }
                                    _ => {}
                                }
                            }
                            return ApiError::Validation(ValidationError::Invalid {
                                field: field.to_string(),
                                message: e.to_string(),
                            });
                        }
                        ApiError::Validation(ValidationError::Invalid {
                            field: "unknown".to_string(),
                            message: e.to_string(),
                        })
                    })?;
                    Ok(obj)
                }
            }
        } else {
            quote! {
                pub fn build(&self) -> #name {
                    self.build_inner().expect("Required fields must be set")
                }
            }
        };

        quote! {
            #doc_attr
            #derives
            #deny_unknown
            #[builder(setter(into, strip_option), build_fn(name = "build_inner", vis = "pub(crate)"))]
            pub struct #name {
                #(#fields)*
                #additional_properties_field
            }

            impl #name {
                pub fn builder() -> #builder_name {
                    #builder_name::default()
                }
            }

            impl #builder_name {
                #build_method
            }
        }
    }
}

impl EnumIr {
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        let name = to_ident(self.name.as_str());
        let name_str = self.name.as_str();
        let derives = emit_derives_for(self);
        let doc_attr = emit_doc(self.description.as_deref());
        let rename_all_attr = self
            .rename_all
            .as_ref()
            .map(|r| quote! { #[serde(rename_all = #r)] });

        let variants = self.variants.iter().enumerate().map(|(i, v)| {
            let v_name = to_ident(&v.rust_name);
            let value = &v.value;
            let v_doc_attr = emit_doc(v.description.as_deref());
            let default_attr = if i == 0 {
                quote! { #[default] }
            } else {
                TokenStream::new()
            };

            let rename_attr = if self.rename_all.is_some() {
                TokenStream::new()
            } else {
                quote! { #[serde(rename = #value)] }
            };

            quote! {
                #v_doc_attr
                #default_attr
                #rename_attr
                #v_name,
            }
        });
        let parse_arms = self.variants.iter().map(|v| {
            let v_ident = to_ident(&v.rust_name);
            let pascal_name = &v.rust_name;
            let value_str = &v.value;
            quote! {
                #value_str | #pascal_name => Ok(Self::#v_ident),
            }
        });
        quote! {
            #doc_attr
            #derives
            #rename_all_attr
            pub enum #name {
                #(#variants)*
            }

            impl std::str::FromStr for #name {
                type Err = String;
                fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                    match s {
                        #(#parse_arms)*
                        _ => Err(format!("unknown variant `{s}` for enum {}", #name_str)),
                    }
                }
            }

            impl From<&str> for #name {
                fn from(s: &str) -> Self {
                    s.parse().unwrap_or_else(|_| {
                        panic!("invalid value for {}: {}", #name_str, s)
                    })
                }
            }

            impl From<String> for #name {
                fn from(s: String) -> Self {
                    Self::from(s.as_str())
                }
            }
        }
    }
}

impl AliasIr {
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        let name = to_ident(self.name.as_str());
        let target = self.target.to_tokens(true);
        let doc_attr = emit_doc(self.description.as_deref());
        quote! {
            #doc_attr
            pub type #name = #target;
        }
    }
}

impl NewtypeIr {
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        let name = to_ident(self.name.as_str());
        let derives = emit_derives_for(self);
        let target = self.target.to_tokens(true);
        let doc_attr = emit_doc(self.description.as_deref());
        let from_str = if matches!(self.target, TypeIr::Primitive(PrimitiveType::String)) {
            quote! {
                impl From<&str> for #name {
                    fn from(s: &str) -> Self {
                        Self(s.to_owned())
                    }
                }
            }
        } else {
            TokenStream::new()
        };
        quote! {
            #doc_attr
            #derives
            pub struct #name(pub #target);
            #from_str
        }
    }
}

impl AnyOfIr {
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        let name = to_ident(self.name.as_str());
        let derives = emit_derives_for(self);
        let doc_attr = emit_doc(self.description.as_deref());

        let variants = self.variants.iter().map(|v| {
            let v_name = to_ident(&v.name);
            let v_type = v.type_info.to_tokens(true);
            let v_doc_attr = emit_doc(None);
            quote! {
                #v_doc_attr
                #[serde(untagged)]
                #v_name(#v_type),
            }
        });

        let from_impls = self.variants.iter().map(|v| {
            let v_name = to_ident(&v.name);
            let v_type = v.type_info.to_tokens(true);
            quote! {
                impl From<#v_type> for #name {
                    fn from(v: #v_type) -> Self {
                        Self::#v_name(v)
                    }
                }
            }
        });

        let first_variant = self.variants.first();
        let first_variant_name =
            first_variant.map_or_else(|| to_ident("Variant0"), |v| to_ident(&v.name));
        let first_variant_type = first_variant.map_or_else(
            || quote! { serde_json::Value },
            |v| v.type_info.to_tokens(true),
        );

        quote! {
            #doc_attr
            #derives
            pub enum #name {
                #(#variants)*
            }

            impl Default for #name {
                fn default() -> Self {
                    Self::#first_variant_name(#first_variant_type::default())
                }
            }

            #(#from_impls)*
        }
    }
}
