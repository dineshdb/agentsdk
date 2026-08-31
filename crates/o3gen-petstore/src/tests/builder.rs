use crate::types::{Pet, PetStatus, Species};
use chrono::Utc;

#[test]
fn test_pet_category_builder_no_unwrap() {
    use crate::types::PetCategory;
    // PetCategory has no validation, so build() should return PetCategory directly
    let category = PetCategory::builder().id("1").name("Dogs").build();

    assert_eq!(category.id, "1");
    assert_eq!(category.name, "Dogs");
}

#[test]
fn test_pet_builder_validation() {
    // Valid pet
    let pet = Pet::builder()
        .id("123")
        .name("Fido")
        .species(Species::Dog)
        .status(PetStatus::Available)
        .age_months(24)
        .price("100.00")
        .currency("USD")
        .created_at(Utc::now())
        .updated_at(Utc::now())
        .build()
        .unwrap();

    assert_eq!(pet.name, "Fido");
    assert_eq!(pet.species, Species::Dog);

    // Invalid pet (name too short)
    let result = Pet::builder()
        .id("123")
        .name("") // invalid: too short (min = 1 in petstore.json)
        .species(Species::Dog)
        .status(PetStatus::Available)
        .age_months(24)
        .price("100.00")
        .currency("USD")
        .created_at(Utc::now())
        .updated_at(Utc::now())
        .build();

    assert!(
        result.is_err(),
        "Builder should fail for invalid pet (empty name)"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("name"),
        "Error should mention 'name'"
    );
}
