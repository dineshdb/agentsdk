use std::collections::HashMap;

use heck::ToSnakeCase;
use http::{Method, StatusCode};
use indexmap::IndexMap;

#[derive(Debug, Clone)]
pub struct ApiIr {
    pub types: IndexMap<String, TypeDefinitionIr>,
    pub operations: Vec<OperationIr>,
    pub security_schemes: Vec<SecuritySchemeIr>,
}

#[derive(Debug, Clone)]
pub enum TypeDefinitionIr {
    Struct(StructIr),
    Enum(EnumIr),
    Alias(AliasIr),
    AnyOf(AnyOfIr),
    Newtype(NewtypeIr),
}

pub trait IrType {
    fn name(&self) -> &str;
    fn set_name(&mut self, name: String);
    fn is_generated(&self) -> bool;
    fn description(&self) -> Option<&str>;
    fn update_references(&mut self, renames: &HashMap<String, String>);
    fn derives(&self) -> &[String];
    fn derives_mut(&mut self) -> &mut Vec<String>;

    fn add_derive(&mut self, trait_name: String) {
        let derives = self.derives_mut();
        if !derives.contains(&trait_name) {
            derives.push(trait_name);
        }
    }

    fn has_derive(&self, trait_name: &str) -> bool {
        self.derives().iter().any(|d| d == trait_name)
    }

    fn remove_derive(&mut self, trait_name: &str) {
        self.derives_mut().retain(|d| d != trait_name);
    }

    fn can_derive_default(&self) -> bool {
        true
    }

    fn has_custom_default(&self) -> bool {
        false
    }

    fn should_derive_default(&self) -> bool {
        self.can_derive_default() && !self.has_custom_default()
    }

    fn should_derive(&self, trait_name: &str) -> bool {
        if trait_name == "Default" {
            return self.should_derive_default();
        }
        true
    }
}

impl IrType for TypeDefinitionIr {
    fn name(&self) -> &str {
        match self {
            Self::Struct(s) => s.name(),
            Self::Enum(e) => e.name(),
            Self::Alias(a) => a.name(),
            Self::AnyOf(a) => a.name(),
            Self::Newtype(n) => n.name(),
        }
    }

    fn set_name(&mut self, name: String) {
        match self {
            Self::Struct(s) => s.set_name(name),
            Self::Enum(e) => e.set_name(name),
            Self::Alias(a) => a.set_name(name),
            Self::AnyOf(a) => a.set_name(name),
            Self::Newtype(n) => n.set_name(name),
        }
    }

    fn is_generated(&self) -> bool {
        match self {
            Self::Struct(s) => s.is_generated(),
            Self::Enum(e) => e.is_generated(),
            Self::Alias(a) => a.is_generated(),
            Self::AnyOf(a) => a.is_generated(),
            Self::Newtype(n) => n.is_generated(),
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Self::Struct(s) => s.description(),
            Self::Enum(e) => e.description(),
            Self::Alias(a) => a.description(),
            Self::AnyOf(a) => a.description(),
            Self::Newtype(n) => n.description(),
        }
    }

    fn update_references(&mut self, renames: &HashMap<String, String>) {
        match self {
            Self::Struct(s) => s.update_references(renames),
            Self::Enum(e) => e.update_references(renames),
            Self::Alias(a) => a.update_references(renames),
            Self::AnyOf(a) => a.update_references(renames),
            Self::Newtype(n) => n.update_references(renames),
        }
    }

    fn derives(&self) -> &[String] {
        match self {
            Self::Struct(s) => s.derives(),
            Self::Enum(e) => e.derives(),
            Self::Alias(a) => a.derives(),
            Self::AnyOf(a) => a.derives(),
            Self::Newtype(n) => n.derives(),
        }
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        match self {
            Self::Struct(s) => s.derives_mut(),
            Self::Enum(e) => e.derives_mut(),
            Self::Alias(a) => a.derives_mut(),
            Self::AnyOf(a) => a.derives_mut(),
            Self::Newtype(n) => n.derives_mut(),
        }
    }

    fn can_derive_default(&self) -> bool {
        match self {
            Self::Struct(s) => s.can_derive_default(),
            Self::Enum(e) => e.can_derive_default(),
            Self::Alias(a) => a.can_derive_default(),
            Self::AnyOf(a) => a.can_derive_default(),
            Self::Newtype(n) => n.can_derive_default(),
        }
    }

    fn has_custom_default(&self) -> bool {
        match self {
            Self::Struct(s) => s.has_custom_default(),
            Self::Enum(e) => e.has_custom_default(),
            Self::Alias(a) => a.has_custom_default(),
            Self::AnyOf(a) => a.has_custom_default(),
            Self::Newtype(n) => n.has_custom_default(),
        }
    }

    fn should_derive(&self, trait_name: &str) -> bool {
        match self {
            Self::Struct(s) => s.should_derive(trait_name),
            Self::Enum(e) => e.should_derive(trait_name),
            Self::Alias(a) => a.should_derive(trait_name),
            Self::AnyOf(a) => a.should_derive(trait_name),
            Self::Newtype(n) => n.should_derive(trait_name),
        }
    }
}

impl TypeDefinitionIr {
    #[must_use]
    pub fn name(&self) -> &str {
        IrType::name(self)
    }

    pub fn set_name(&mut self, name: String) {
        IrType::set_name(self, name);
    }

    #[must_use]
    pub fn is_generated(&self) -> bool {
        IrType::is_generated(self)
    }

    pub fn update_references(&mut self, renames: &HashMap<String, String>) {
        IrType::update_references(self, renames);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Name {
    Provided(String),
    Generated(String),
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Self::Provided(s)
    }
}

impl Name {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Provided(s) | Self::Generated(s) => s,
        }
    }

    pub fn set_string(&mut self, new_name: String) {
        match self {
            Self::Provided(s) | Self::Generated(s) => *s = new_name,
        }
    }

    #[must_use]
    pub fn is_generated(&self) -> bool {
        matches!(self, Self::Generated(_))
    }
}

#[derive(Debug, Clone)]
pub struct StructIr {
    pub name: Name,
    pub fields: Vec<FieldIr>,
    pub derives: Vec<String>,
    pub description: Option<String>,
    pub additional_properties_type: Option<TypeIr>,
}

impl StructIr {
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn set_name(&mut self, name: String) {
        self.name.set_string(name);
    }

    #[must_use]
    pub fn is_generated(&self) -> bool {
        self.name.is_generated()
    }

    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn update_references(&mut self, renames: &HashMap<String, String>) {
        for field in &mut self.fields {
            field.type_info.update_reference(renames);
        }
        if let Some(ap) = &mut self.additional_properties_type {
            ap.update_reference(renames);
        }
    }

    #[must_use]
    pub fn derives(&self) -> &[String] {
        &self.derives
    }

    pub fn derives_mut(&mut self) -> &mut Vec<String> {
        &mut self.derives
    }

    #[must_use]
    pub fn can_derive_default(&self) -> bool {
        self.fields
            .iter()
            .all(|f| f.type_info.can_derive_default() || !f.required)
    }

    #[must_use]
    pub fn has_validation(&self) -> bool {
        self.fields.iter().any(|f| !f.validation.is_empty())
    }

    #[must_use]
    pub fn is_infallible(&self) -> bool {
        !self.has_validation() && self.fields.iter().all(|f| !f.required)
    }
}

impl IrType for StructIr {
    fn name(&self) -> &str {
        self.name()
    }

    fn set_name(&mut self, name: String) {
        self.set_name(name);
    }

    fn is_generated(&self) -> bool {
        self.is_generated()
    }

    fn description(&self) -> Option<&str> {
        self.description()
    }

    fn update_references(&mut self, renames: &HashMap<String, String>) {
        self.update_references(renames);
    }

    fn derives(&self) -> &[String] {
        self.derives()
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        self.derives_mut()
    }

    fn can_derive_default(&self) -> bool {
        self.can_derive_default()
    }
}

#[derive(Debug, Clone)]
pub struct EnumIr {
    pub name: Name,
    pub variants: Vec<EnumVariantIr>,
    pub derives: Vec<String>,
    pub rename_all: Option<String>,
    pub description: Option<String>,
}

impl IrType for EnumIr {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn set_name(&mut self, name: String) {
        self.name.set_string(name);
    }

    fn is_generated(&self) -> bool {
        self.name.is_generated()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn update_references(&mut self, _renames: &HashMap<String, String>) {}

    fn derives(&self) -> &[String] {
        &self.derives
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        &mut self.derives
    }
}

#[derive(Debug, Clone)]
pub struct NewtypeIr {
    pub name: Name,
    pub target: TypeIr,
    pub derives: Vec<String>,
    pub description: Option<String>,
}

impl IrType for NewtypeIr {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn set_name(&mut self, name: String) {
        self.name.set_string(name);
    }

    fn is_generated(&self) -> bool {
        self.name.is_generated()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn update_references(&mut self, renames: &HashMap<String, String>) {
        self.target.update_reference(renames);
    }

    fn derives(&self) -> &[String] {
        &self.derives
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        &mut self.derives
    }

    fn can_derive_default(&self) -> bool {
        self.target.can_derive_default()
    }
}

#[derive(Debug, Clone)]
pub struct FieldIr {
    pub name: String,      // Original name in JSON
    pub rust_name: String, // Snake case identifier
    pub type_info: TypeIr,
    pub required: bool,
    pub validation: Vec<ValidationIr>,
    pub serde_rename: Option<String>,
    pub description: Option<String>,
}

impl FieldIr {
    #[must_use]
    pub fn new(
        name: &str,
        type_info: TypeIr,
        required: bool,
        validation: Vec<ValidationIr>,
        description: Option<String>,
    ) -> Self {
        let rust_name = name.to_snake_case();
        Self {
            name: name.to_string(),
            serde_rename: if name == rust_name {
                None
            } else {
                Some(name.to_string())
            },
            rust_name,
            type_info,
            required,
            validation,
            description,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnumVariantIr {
    pub name: String,
    pub rust_name: String,
    pub value: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AliasIr {
    pub name: Name,
    pub target: TypeIr,
    pub description: Option<String>,
    pub derives: Vec<String>,
}

impl IrType for AliasIr {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn set_name(&mut self, name: String) {
        self.name.set_string(name);
    }

    fn is_generated(&self) -> bool {
        self.name.is_generated()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn update_references(&mut self, renames: &HashMap<String, String>) {
        self.target.update_reference(renames);
    }

    fn derives(&self) -> &[String] {
        &self.derives
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        &mut self.derives
    }
}

#[derive(Debug, Clone)]
pub struct AnyOfIr {
    pub name: Name,
    pub variants: Vec<VariantIr>,
    pub derives: Vec<String>,
    pub description: Option<String>,
}

impl IrType for AnyOfIr {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn set_name(&mut self, name: String) {
        self.name.set_string(name);
    }

    fn is_generated(&self) -> bool {
        self.name.is_generated()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn update_references(&mut self, renames: &HashMap<String, String>) {
        for variant in &mut self.variants {
            variant.type_info.update_reference(renames);
        }
    }

    fn derives(&self) -> &[String] {
        &self.derives
    }

    fn derives_mut(&mut self) -> &mut Vec<String> {
        &mut self.derives
    }

    fn has_custom_default(&self) -> bool {
        true
    }

    fn should_derive(&self, trait_name: &str) -> bool {
        if trait_name == "Default" || trait_name == "derive_more::From" {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct VariantIr {
    pub name: String,
    pub type_info: TypeIr,
}

#[derive(Debug, Clone)]
pub enum TypeIr {
    Reference(String),
    Primitive(PrimitiveType),
    Array(Box<TypeIr>),
    Map(Box<TypeIr>),
    Value,        // serde_json::Value
    Enum(String), // Reference to an enum type definition
}

impl From<PrimitiveType> for TypeIr {
    fn from(p: PrimitiveType) -> Self {
        Self::Primitive(p)
    }
}

impl TypeIr {
    #[must_use]
    pub fn can_derive_default(&self) -> bool {
        match self {
            Self::Reference(_)
            | Self::Enum(_)
            | Self::Primitive(_)
            | Self::Array(_)
            | Self::Map(_)
            | Self::Value => true,
        }
    }

    pub fn update_reference(&mut self, renames: &HashMap<String, String>) {
        match self {
            Self::Reference(name) | Self::Enum(name) => {
                if let Some(new_name) = renames.get(name) {
                    *name = new_name.clone();
                }
            }
            Self::Array(inner) | Self::Map(inner) => inner.update_reference(renames),
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    String,
    Integer,
    Number,
    Boolean,
    Date,
    DateTime,
}

#[derive(Debug, Clone, Copy)]
pub enum ValidationIr {
    Length { min: Option<u64>, max: Option<u64> },
    FloatRange { min: Option<f64>, max: Option<f64> },
    IntRange { min: Option<i64>, max: Option<i64> },
}

#[derive(Debug, Clone)]
pub struct OperationIr {
    pub operation_id: String,
    pub method: Method,
    pub path: String,
    pub parameters: Vec<ParameterIr>,
    pub request_body: Option<TypeIr>,
    pub responses: Vec<ResponseIr>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParameterIr {
    pub name: String,
    pub location: ParameterLocation,
    pub required: bool,
    pub type_info: TypeIr,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterLocation {
    Path,
    Query,
    Header,
    Cookie,
}

#[derive(Debug, Clone)]
pub struct ResponseIr {
    pub code: StatusCode,
    pub type_info: Option<TypeIr>,
}

#[derive(Debug, Clone)]
pub enum SecuritySchemeIr {
    HttpBearer,
    HttpBasic,
    ApiKey {
        location: ApiKeyLocation,
        field_name: String,
    },
    OAuth2,
    OpenIdConnect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKeyLocation {
    Query,
    Header,
    Cookie,
}
