use crate::types;

#[test]
fn test_compare_pets() {
    let pet1 = types::Pet {
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

    let pet2 = types::Pet {
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

    assert_eq!(pet1, pet2);
}

#[test]
fn test_compare_different_pets() {
    let pet1 = types::Pet {
        id: "1".to_string(),
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

    let pet2 = types::Pet {
        id: "2".to_string(),
        name: "cat".to_string(),
        species: types::Species::Cat,
        status: types::PetStatus::Available,
        price: "50.00".to_string(),
        currency: "USD".to_string(),
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        age_months: 24,
        ..Default::default()
    };

    assert_ne!(pet1, pet2);
}

#[test]
fn test_cloning_pet() {
    let original = types::Pet {
        id: "1".to_string(),
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

    let cloned = original.clone();
    assert_eq!(original, cloned);
}
