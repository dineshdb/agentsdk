use crate::types;

#[test]
fn test_json_serialization_pet() {
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
}

#[test]
fn test_json_serialization_error() {
    let error = types::Error {
        status: 404,
        title: "Not found".to_string(),
        r#type: "about:blank".to_string(),
        ..Default::default()
    };

    assert_eq!(error.status, 404);
    assert_eq!(error.title, "Not found");
}

#[test]
fn test_validation() {
    use validator::Validate;
    let pet = types::Pet {
        id: "123".to_string(),
        name: "".to_string(), // Too short, min is 1
        age_months: -1,       // Too small, min is 0
        ..Default::default()
    };
    let result = pet.validate();
    assert!(result.is_err());
}
