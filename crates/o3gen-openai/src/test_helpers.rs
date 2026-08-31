pub mod mock {
    use crate::OpenAIApiClient;
    use mockito::{Mock, Server, ServerGuard};
    use serde::Serialize;

    pub struct MockServer {
        pub server: ServerGuard,
    }

    impl MockServer {
        pub async fn new() -> Self {
            Self {
                server: Server::new_async().await,
            }
        }

        pub fn url(&self) -> String {
            self.server.url()
        }

        pub fn client(&self) -> OpenAIApiClient {
            OpenAIApiClient::new(self.url())
        }

        pub fn client_with_key(&self, api_key: &str) -> OpenAIApiClient {
            OpenAIApiClient::new(self.url()).with_api_key(api_key.into())
        }

        pub fn mock_get(&mut self, path: &str, status: usize, body: &impl Serialize) -> Mock {
            self.server
                .mock("GET", path)
                .with_status(status)
                .with_body(serde_json::to_string(body).unwrap())
                .create()
        }

        pub fn mock_post(&mut self, path: &str, status: usize, body: &impl Serialize) -> Mock {
            self.server
                .mock("POST", path)
                .with_status(status)
                .with_body(serde_json::to_string(body).unwrap())
                .create()
        }

        pub fn mock_delete(&mut self, path: &str, status: usize, body: &impl Serialize) -> Mock {
            self.server
                .mock("DELETE", path)
                .with_status(status)
                .with_body(serde_json::to_string(body).unwrap())
                .create()
        }
    }
}
