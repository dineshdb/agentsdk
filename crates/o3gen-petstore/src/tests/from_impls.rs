use std::str::FromStr;

use crate::types as petstore_types;

include!(concat!(env!("OUT_DIR"), "/newtypes.rs"));

#[test]
fn test_from_str_ref_for_pet_status() {
    let status = petstore_types::PetStatus::from("Available");
    assert_eq!(status, petstore_types::PetStatus::Available);

    let status: petstore_types::PetStatus = "Pending".into();
    assert_eq!(status, petstore_types::PetStatus::Pending);

    let status: petstore_types::PetStatus = "Adopted".into();
    assert_eq!(status, petstore_types::PetStatus::Adopted);

    let status: petstore_types::PetStatus = "NotAvailable".into();
    assert_eq!(status, petstore_types::PetStatus::NotAvailable);
}

#[test]
fn test_from_str_ref_matches_spec_value() {
    let status: petstore_types::PetStatus = "AVAILABLE".into();
    assert_eq!(status, petstore_types::PetStatus::Available);

    let status: petstore_types::PetStatus = "PENDING".into();
    assert_eq!(status, petstore_types::PetStatus::Pending);
}

#[test]
fn test_from_str_ref_for_species() {
    let species: petstore_types::Species = "Dog".into();
    assert_eq!(species, petstore_types::Species::Dog);

    let species: petstore_types::Species = "CAT".into();
    assert_eq!(species, petstore_types::Species::Cat);
}

#[test]
fn test_from_str_ref_panics_on_invalid() {
    let result = std::panic::catch_unwind(|| {
        let _: petstore_types::PetStatus = "Bogus".into();
    });
    assert!(result.is_err());
}

// --- From<String> ---

#[test]
fn test_from_string_for_pet_status() {
    let status = petstore_types::PetStatus::from("Available".to_string());
    assert_eq!(status, petstore_types::PetStatus::Available);

    let status: petstore_types::PetStatus = "Pending".to_string().into();
    assert_eq!(status, petstore_types::PetStatus::Pending);
}

#[test]
fn test_from_string_in_function_arg() {
    fn accept_status(_s: petstore_types::PetStatus) {}
    accept_status("Available".to_string().into());
    accept_status("NotAvailable".to_string().into());
}

// --- FromStr ---

#[test]
fn test_from_str_trait_with_rust_name() {
    let status = petstore_types::PetStatus::from_str("Available").unwrap();
    assert_eq!(status, petstore_types::PetStatus::Available);
}

#[test]
fn test_from_str_trait_with_spec_value() {
    let status = petstore_types::PetStatus::from_str("AVAILABLE").unwrap();
    assert_eq!(status, petstore_types::PetStatus::Available);
}

#[test]
fn test_from_str_trait_via_parse() {
    let status: petstore_types::PetStatus = "Pending".parse().unwrap();
    assert_eq!(status, petstore_types::PetStatus::Pending);

    let status: petstore_types::PetStatus = "ADOPTED".parse().unwrap();
    assert_eq!(status, petstore_types::PetStatus::Adopted);
}

#[test]
fn test_from_str_trait_error() {
    let result = petstore_types::PetStatus::from_str("Bogus");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Bogus"));
}

#[test]
fn test_from_str_trait_for_order_status() {
    let status = petstore_types::OrderStatus::from_str("Placed").unwrap();
    assert_eq!(status, petstore_types::OrderStatus::Placed);

    let status = petstore_types::OrderStatus::from_str("SHIPPED").unwrap();
    assert_eq!(status, petstore_types::OrderStatus::Shipped);
}

// --- From<&str> for newtypes ---

#[test]
fn test_from_str_for_string_newtype() {
    let id = PetId::from("my-pet-id");
    assert_eq!(id, PetId("my-pet-id".to_string()));

    let id: PetId = "pet-123".into();
    assert_eq!(id.0, "pet-123");
}

#[test]
fn test_multiple_newtypes() {
    let pet_id: PetId = "pet-1".into();
    let item_id: ItemId = "item-1".into();
    assert_ne!(pet_id, PetId("item-1".to_string()));
    assert_eq!(pet_id, PetId("pet-1".to_string()));
    assert_eq!(item_id, ItemId("item-1".to_string()));
}
