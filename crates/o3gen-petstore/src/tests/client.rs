use crate::PetApi;
use crate::PetApiClient;
use crate::types::{
    ListPetsParams, ListPetsSpecies, Pagination, Pet, PetCollection, PetStatus, Species,
};
use mockito::Server;

#[tokio::test]
async fn test_list_pets() {
    let mut server = Server::new_async().await;
    let url = server.url();
    let client = PetApiClient::new(url);

    let mock_pet = Pet {
        id: "1".to_string(),
        name: "Fido".to_string(),
        age_months: 12,
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        currency: "USD".to_string(),
        price: "100.00".to_string(),
        species: Species::Dog,
        status: PetStatus::Available,
        ..Default::default()
    };

    let mock_pets = PetCollection {
        data: vec![mock_pet],
        pagination: Pagination {
            total_items: 1,
            total_pages: 1,
            limit: 10,
            page: 1,
        },
    };

    let _m = server
        .mock("GET", mockito::Matcher::Regex("^/pets".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_string(&mock_pets).unwrap())
        .create_async()
        .await;

    let query = ListPetsParams {
        species: Some(ListPetsSpecies::Dog),
        page: Some(1),
        limit: Some(10),
        ..Default::default()
    };

    let result = client.list_pets(query).await.unwrap();
    assert_eq!(result.data.len(), 1);
    assert_eq!(result.data[0].name, "Fido");
}

#[tokio::test]
async fn test_get_pet() {
    let mut server = Server::new_async().await;
    let url = server.url();
    let client = PetApiClient::new(url);

    let mock_pet = Pet {
        id: "123".to_string(),
        name: "Buddy".to_string(),
        age_months: 24,
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        currency: "USD".to_string(),
        price: "200.00".to_string(),
        species: Species::Dog,
        status: PetStatus::Available,
        ..Default::default()
    };

    let _m = server
        .mock("GET", "/pets/123")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_string(&mock_pet).unwrap())
        .create_async()
        .await;

    let result = client.get_pet("123".to_string()).await.unwrap();
    assert_eq!(result.id, "123");
    assert_eq!(result.name, "Buddy");
}

#[tokio::test]
async fn test_api_error() {
    let mut server = Server::new_async().await;
    let url = server.url();
    let client = PetApiClient::new(url);

    let _m = server
        .mock("GET", mockito::Matcher::Regex("^/pets".to_string()))
        .with_status(404)
        .create_async()
        .await;

    let result = client.list_pets(ListPetsParams::default()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_with_api_key_sends_bearer_token() {
    let mut server = Server::new_async().await;
    let url = server.url();
    let client = PetApiClient::new(url).with_api_key("sk-test-key".into());

    let mock_pet = Pet {
        id: "1".to_string(),
        name: "Fido".to_string(),
        age_months: 12,
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        currency: "USD".to_string(),
        price: "100.00".to_string(),
        species: Species::Dog,
        status: PetStatus::Available,
        ..Default::default()
    };

    let mock_pets = PetCollection {
        data: vec![mock_pet],
        pagination: Pagination {
            total_items: 1,
            total_pages: 1,
            limit: 10,
            page: 1,
        },
    };

    let _m = server
        .mock("GET", mockito::Matcher::Regex("^/pets".to_string()))
        .match_header("Authorization", "Bearer sk-test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_string(&mock_pets).unwrap())
        .create_async()
        .await;

    let query = ListPetsParams {
        species: Some(ListPetsSpecies::Dog),
        page: Some(1),
        limit: Some(10),
        ..Default::default()
    };

    let result = client.list_pets(query).await.unwrap();
    assert_eq!(result.data.len(), 1);
    assert_eq!(result.data[0].name, "Fido");
}

#[tokio::test]
async fn test_without_api_key_omits_auth_header() {
    let mut server = Server::new_async().await;
    let url = server.url();
    let client = PetApiClient::new(url);

    let mock_pet = Pet {
        id: "1".to_string(),
        name: "Fido".to_string(),
        age_months: 12,
        created_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        updated_at: chrono::DateTime::from_timestamp(1736412000, 0).unwrap(),
        currency: "USD".to_string(),
        price: "100.00".to_string(),
        species: Species::Dog,
        status: PetStatus::Available,
        ..Default::default()
    };

    let mock_pets = PetCollection {
        data: vec![mock_pet],
        pagination: Pagination {
            total_items: 1,
            total_pages: 1,
            limit: 10,
            page: 1,
        },
    };

    let _m = server
        .mock("GET", mockito::Matcher::Regex("^/pets".to_string()))
        .match_header("Authorization", mockito::Matcher::Missing)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_string(&mock_pets).unwrap())
        .create_async()
        .await;

    let query = ListPetsParams {
        species: Some(ListPetsSpecies::Dog),
        page: Some(1),
        limit: Some(10),
        ..Default::default()
    };

    let result = client.list_pets(query).await.unwrap();
    assert_eq!(result.data.len(), 1);
    assert_eq!(result.data[0].name, "Fido");
}
