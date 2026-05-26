#![deny(missing_docs)]
//! Macros for the `agentsdk` library.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{
    Expr, ExprLit, FnArg, ItemFn, Lit, Meta, MetaNameValue, Pat, PatType, Path, Token, Type,
    parse_macro_input, punctuated::Punctuated,
};

#[proc_macro_attribute]
/// Constructs a tool from a function defnition. A tool has a name, a description,
/// an input and a body. all three components are infered from a standard rust
/// function. The name is the defined name of the function,
/// The description is infered from the doc comments of the function, The input
/// infered from the function arguments.
///
/// # Example
///
/// ```ignore
/// use agentsdk::tool;
/// use agentsdk::core::tools::Tool;
///
/// #[tool]
/// /// Returns the username
/// fn get_username(id: String) -> Tool {
///     Ok(format!("user_{}", id))
/// }
/// ```
///
/// - `get_username` becomes the name of the tool
/// - `"Returns the username"` becomes the description of the tool
/// - `id: String` becomes the input of the tool. converted to `{"id": "string"}`
///   as json schema
///
/// The function should return a `Result<String, String>` eventhough the return statement
/// returns a `Tool` object. This is because the macro will automatically convert the
/// function into a `Tool` object and return it. You should return what the model can
/// understand as a `String`.
///
/// In the event that the model refuses to send an argument, the default implementation
/// will be used. this works perfectly for arguments that are `Option`s. Make sure to
/// use `Option` types for arguments that are optional or implement a default for those
/// that are not and handle those defaults accordingly in the tool body.
///
/// A single parameter typed as `ToolContext` is treated as runtime context and is not
/// included in the schema sent to the model.
///
/// You can override name and description using the macro arguments `name` and `desc`.
///
/// # Example with overrides
/// ```ignore
/// use agentsdk::tool;
/// use agentsdk::core::tools::Tool;
///
///     #[tool(
///         name = "the-name-for-this-tool",
///         desc = "the-description-for-this-tool"
///     )]
///     fn get_username(id: String) -> Tool {
///         Ok(format!("user_{}", id))
///     }
/// ```
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;
    let return_type = &input_fn.sig.output;
    let is_async = input_fn.sig.asyncness.is_some();
    let block = &input_fn.block;
    let inputs = &input_fn.sig.inputs;
    let attrs = &input_fn.attrs;
    let args_parser = Punctuated::<MetaNameValue, Token![,]>::parse_terminated;
    let args = args_parser.parse(attr);

    let (name_arg, description_arg) = if let Ok(args) = args {
        let mut name: Option<String> = None;
        let mut description: Option<String> = None;

        for arg in args {
            if arg.path.is_ident("desc")
                && let Expr::Lit(lit) = &arg.value
                && let Lit::Str(str_lit) = &lit.lit
            {
                description = Some(str_lit.value());
            } else if arg.path.is_ident("name")
                && let Expr::Lit(lit) = &arg.value
                && let Lit::Str(str_lit) = &lit.lit
            {
                name = Some(str_lit.value());
            }
        }

        (name, description)
    } else {
        (None, None)
    };

    let description = if let Some(desc) = description_arg {
        desc
    } else {
        // Extract doc comments
        let doc_comments: Vec<String> = attrs
            .iter()
            .filter_map(|attr| {
                if attr.path().is_ident("doc") {
                    if let Meta::NameValue(meta_name_value) = &attr.meta {
                        if let Expr::Lit(ExprLit {
                            lit: Lit::Str(lit_str),
                            ..
                        }) = &meta_name_value.value
                        {
                            let doc = lit_str.value();
                            let doc = doc.strip_prefix(' ').unwrap_or(&doc).to_string();
                            Some(doc)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        doc_comments.join("\n")
    };

    let name = if let Some(name) = name_arg {
        name
    } else {
        fn_name.to_string()
    };

    let mut context_count = 0usize;
    let mut binding_tokens = Vec::new();
    let mut struct_fields = Vec::new();

    for pat_type in inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(pat_type) => Some(pat_type),
        FnArg::Receiver(_) => None,
    }) {
        let (ident, ty, is_context) = match parse_tool_parameter(pat_type) {
            Ok(parameter) => parameter,
            Err(error) => return error.to_compile_error().into(),
        };

        if is_context {
            context_count += 1;
            if context_count > 1 {
                return syn::Error::new_spanned(
                    pat_type,
                    "only one ToolContext parameter is supported",
                )
                .to_compile_error()
                .into();
            }

            binding_tokens.push(quote! {
                let #ident: #ty = _ctx.clone();
            });
            continue;
        }

        let ident_str = ident.to_string();
        binding_tokens.push(quote! {
            let #ident: #ty = inp.as_object()
                .and_then(|obj| obj.get(#ident_str))
                .and_then(|val| ::agentsdk::__private::serde_json::from_value(val.clone()).ok())
                .unwrap_or_default(); // use default value if model doesn't send arg or it's invalid
        });
        struct_fields.push(quote! { #ident: #ty });
    }

    let execute_impl = if is_async {
        quote! {
            ::agentsdk::core::tools::ToolExecute::from_async(|_ctx, inp| async move {
                #(#binding_tokens)*
                let res = (async move { #block }).await;
                res.map(|out| ::agentsdk::__private::serde_json::to_value(out).unwrap())
                   .map_err(|e| e.to_string())
            })
        }
    } else {
        quote! {
            ::agentsdk::core::tools::ToolExecute::from_sync(|_ctx, inp| {
                #(#binding_tokens)*
                let res = (|| { #block })();
                res.map(|out| ::agentsdk::__private::serde_json::to_value(out).unwrap())
                   .map_err(|e| e.to_string())
            })
        }
    };

    let expanded = quote! {
        #vis fn #fn_name() #return_type  {
            use ::agentsdk::__private::schemars::{schema_for, JsonSchema, Schema};

            #[derive(::agentsdk::__private::schemars::JsonSchema, Debug)]
            #[schemars(crate = "::agentsdk::__private::schemars")]
            #[allow(dead_code)]
            struct Function {
                #(#struct_fields),*
            }

            let input_schema = schema_for!(Function);

            let definition = ::agentsdk::core::tools::ToolDefinition::builder()
                .name(#name.to_string())
                .description(#description.to_string())
                .input_schema(input_schema)
                .build()
                .expect("Failed to build tool definition");

            ::agentsdk::core::tools::Tool::builder()
                .definition(definition)
                .execute(#execute_impl)
                .build()
                .expect("Failed to build tool")
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(PluginTools, attributes(tool))]
/// Derives `PluginTools` for an enum.
pub fn derive_plugin_tools(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let enum_name = &input.ident;

    let data = match &input.data {
        syn::Data::Enum(data) => data,
        _ => {
            return syn::Error::new_spanned(&input, "Only enums supported")
                .to_compile_error()
                .into();
        }
    };

    let (mut definitions, mut from_call_arms) = (Vec::new(), Vec::new());

    for variant in &data.variants {
        let variant_name = &variant.ident;
        let mut tool_name = variant_name.to_string();

        // Parse attributes
        for attr in &variant.attrs {
            if attr.path().is_ident("tool")
                && let Ok(args) =
                    attr.parse_args_with(Punctuated::<MetaNameValue, Token![,]>::parse_terminated)
            {
                for arg in args {
                    if arg.path.is_ident("name")
                        && let Expr::Lit(lit) = &arg.value
                        && let Lit::Str(s) = &lit.lit
                    {
                        tool_name = s.value();
                    }
                }
            }
        }

        // Get description
        let description = variant
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("doc"))
            .filter_map(|a| {
                if let Meta::NameValue(m) = &a.meta {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &m.value
                    {
                        Some(s.value().trim().to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Handle variant payload
        match &variant.fields {
            syn::Fields::Unit => {
                definitions.push(quote! {
                    ::agentsdk::core::tools::ToolDefinition::builder()
                        .name(#tool_name).description(#description)
                        .input_schema(::agentsdk::__private::schemars::schema_for!(::agentsdk::__private::serde_json::Value))
                        .build().expect("definition")
                });
                from_call_arms.push(quote! { #tool_name => Ok(Self::#variant_name) });
            }
            syn::Fields::Unnamed(f) if f.unnamed.len() == 1 => {
                let ty = &f.unnamed[0].ty;
                definitions.push(quote! {
                    ::agentsdk::core::tools::ToolDefinition::builder()
                        .name(#tool_name).description(#description)
                        .input_schema(::agentsdk::__private::schemars::schema_for!(#ty))
                        .build().expect("definition")
                });
                from_call_arms.push(quote! {
                    #tool_name => Ok(Self::#variant_name(::agentsdk::__private::serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?))
                });
            }
            _ => {
                return syn::Error::new_spanned(variant, "Zero or one unnamed field only")
                    .to_compile_error()
                    .into();
            }
        }
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    TokenStream::from(quote! {
        impl #impl_generics ::agentsdk::core::plugin::PluginTools for #enum_name #ty_generics #where_clause {
            fn definitions() -> Vec<::agentsdk::core::tools::ToolDefinition> { vec![ #(#definitions),* ] }
            fn from_call(call: &::agentsdk::core::plugin::PluginToolCall) -> Result<Self, String> {
                match call.name.as_str() {
                    #(#from_call_arms,)*
                    _ => Err(format!("Unknown tool: {}", call.name)),
                }
            }
        }
    })
}

fn parse_tool_parameter(pat_type: &PatType) -> syn::Result<(syn::Ident, Type, bool)> {
    let Pat::Ident(pat_ident) = &*pat_type.pat else {
        return Err(syn::Error::new_spanned(
            &pat_type.pat,
            "#[tool] only supports identifier parameters",
        ));
    };

    let ident = pat_ident.ident.clone();
    let ty = (*pat_type.ty).clone();
    Ok((ident, ty.clone(), is_tool_context_type(&ty)))
}

/// Checks if the given type is a `ToolContext` type.
fn is_tool_context_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };

    let Some(last_segment) = type_path.path.segments.last() else {
        return false;
    };

    if last_segment.ident != "ToolContext"
        || !matches!(last_segment.arguments, syn::PathArguments::None)
    {
        return false;
    }

    is_supported_tool_context_path(&type_path.path)
}

/// Checks if the given path is a supported `ToolContext` path.
fn is_supported_tool_context_path(path: &Path) -> bool {
    let segments: Vec<_> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let segments: Vec<_> = segments.iter().map(String::as_str).collect();

    matches!(
        segments.as_slice(),
        ["ToolContext"]
            | ["tools", "ToolContext"]
            | ["core", "tools", "ToolContext"]
            | ["agentsdk", "core", "tools", "ToolContext"]
    )
}
