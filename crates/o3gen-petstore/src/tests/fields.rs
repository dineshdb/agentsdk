use crate::types;

#[test]
fn test_pet_with_required_fields() {
    let pet = types::Pet {
        id: "123".to_string(),
        name: "doggie".to_string(),
        species: types::Species::Dog,
        status: types::PetStatus::Available,
        price: "100.00".to_string(),
        currency: "USD".to_string(),
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        age_months: 12,
        ..Default::default()
    };
    assert_eq!(pet.id, "123");
    assert_eq!(pet.name, "doggie");
    assert_eq!(pet.age_months, 12);
}

#[test]
fn test_error_with_reserved_keyword_field() {
    let error = types::Error {
        status: 404,
        title: "Not found".to_string(),
        r#type: "about:blank".to_string(),
        ..Default::default()
    };
    assert_eq!(error.status, 404);
    assert_eq!(error.title, "Not found");
    assert_eq!(error.r#type, "about:blank");
}
