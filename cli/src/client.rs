use serde::{de::DeserializeOwned, Serialize};

#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub code: String,
    pub message: String,
    pub request_id: Option<String>,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug)]
pub enum ClientError {
    Api(ApiError),
    Transport(String),
}

impl ClientError {
    pub fn exit_code(&self) -> i32 {
        match self {
            ClientError::Api(_) => 2,
            ClientError::Transport(_) => 1,
        }
    }
}

pub struct RagClient {
    base_url: String,
    user: Option<String>,
}

/// Login sent in `X-User-Login`: `RR_USER`, else the OS user (`USER` / `USERNAME`).
/// A reverse proxy in front of the server may overwrite it; without one the server
/// uses it as the caller identity for its audit log.
pub fn default_user() -> Option<String> {
    ["RR_USER", "USER", "USERNAME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .find(|v| !v.trim().is_empty())
}

impl RagClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_user(base_url, default_user())
    }

    pub fn with_user(base_url: impl Into<String>, user: Option<String>) -> Self {
        Self { base_url: base_url.into().trim_end_matches('/').to_string(), user }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    pub fn post<B: Serialize, T: DeserializeOwned>(
        &self, path: &str, body: &B,
    ) -> Result<T, ClientError> {
        let rid = uuid::Uuid::new_v4().to_string();
        let mut req = ureq::post(&self.url(path))
            .set("Accept", "application/json")
            .set("Content-Type", "application/json")
            .set("X-Request-Id", &rid);
        if let Some(u) = &self.user {
            req = req.set("X-User-Login", u);
        }
        let resp = req.send_json(body);
        Self::handle(resp)
    }

    pub fn get<T: DeserializeOwned>(
        &self, path: &str, query: &[(&str, &str)],
    ) -> Result<T, ClientError> {
        let rid = uuid::Uuid::new_v4().to_string();
        let mut req = ureq::get(&self.url(path))
            .set("Accept", "application/json")
            .set("X-Request-Id", &rid);
        if let Some(u) = &self.user {
            req = req.set("X-User-Login", u);
        }
        for (k, v) in query {
            req = req.query(k, v);
        }
        Self::handle(req.call())
    }

    fn handle<T: DeserializeOwned>(
        resp: Result<ureq::Response, ureq::Error>,
    ) -> Result<T, ClientError> {
        match resp {
            Ok(r) => r.into_json::<T>().map_err(|e| ClientError::Transport(e.to_string())),
            Err(ureq::Error::Status(status, r)) => {
                let body = r.into_string().unwrap_or_default();
                Err(ClientError::Api(parse_error(status, &body)))
            }
            Err(e) => Err(ClientError::Transport(e.to_string())),
        }
    }
}

fn parse_error(status: u16, body: &str) -> ApiError {
    match serde_json::from_str::<crate::model::ErrorEnvelope>(body) {
        Ok(env) => ApiError {
            status,
            code: env.error.code,
            message: env.error.message,
            request_id: env.error.request_id,
            details: env.error.details,
        },
        Err(_) => ApiError {
            status,
            code: "unknown".into(),
            message: if body.is_empty() { format!("HTTP {status}") } else { body.to_string() },
            request_id: None,
            details: None,
        },
    }
}
