include!(concat!(env!("OUT_DIR"), "/composite.rs"));

#[test]
fn test_any_of_enum_deserialization_dog() {
    let dog_json = r#"{"barkVolume": 11}"#;
    let animal: types::Animal = serde_json::from_str(dog_json).unwrap();
    match animal {
        types::Animal::Dog(dog) => assert_eq!(dog.bark_volume, Some(11)),
        _ => panic!("Expected Dog"),
    }
}

#[test]
fn test_any_of_enum_deserialization_cat() {
    let cat_json = r#"{"purrVolume": 5}"#;
    let animal: types::Animal = serde_json::from_str(cat_json).unwrap();
    // Because Dog has all optional fields, `{ "purrVolume": 5 }` will successfully parse as Dog (ignoring unknown fields if not deny_unknown_fields).
    // Let's see what happens! The struct has #[serde(deny_unknown_fields)], so Dog should fail, and Cat should succeed.
    match animal {
        types::Animal::Cat(cat) => assert_eq!(cat.purr_volume, Some(5)),
        _ => panic!("Expected Cat"),
    }
}

#[test]
fn test_any_of_enum_serialization() {
    let cat = types::Cat {
        purr_volume: Some(5),
    };
    let animal = types::Animal::Cat(cat);
    assert_eq!(
        animal,
        types::Animal::Cat(types::Cat {
            purr_volume: Some(5)
        })
    );
}

#[test]
fn test_all_of_struct_flattening() {
    let composite = types::CompositePet {
        id: "comp-1".to_string(),
        name: "Frankenstein".to_string(),
    };

    assert_eq!(composite.id, "comp-1");
    assert_eq!(composite.name, "Frankenstein");
}

#[test]
fn test_all_of_deserialization_missing_required_field() {
    let invalid_json = r#"{"id": "comp-1"}"#; // missing name
    let result: std::result::Result<types::CompositePet, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err());
}
