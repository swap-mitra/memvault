//! Optional embedding provider: an OpenAI-compatible `/embeddings`
//! endpoint the server calls when a caller writes or searches without
//! supplying a vector. An LLM driving an MCP client cannot produce an
//! embedding itself, so without this every MCP user was running
//! keyword-only search whether they knew it or not.
//!
//! Off unless `MEMVAULT_EMBED_URL` and `MEMVAULT_EMBED_MODEL` are both set;
//! the default binary still runs no model and opens no socket. When on, a
//! provider failure is the caller's error, never a silent fall back to
//! keyword search: an agent that believes it has semantic recall and
//! doesn't is worse off than one that knows.

use serde::{Deserialize, Serialize};

pub struct Embedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<Datum>,
}

#[derive(Deserialize)]
struct Datum {
    embedding: Vec<f32>,
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.trim().is_empty())
}

impl Embedder {
    /// `MEMVAULT_EMBED_URL` is the API base (`https://api.openai.com/v1`,
    /// `http://localhost:11434/v1` for Ollama); `/embeddings` is appended.
    /// `MEMVAULT_EMBED_API_KEY`, if set, goes out as a bearer token.
    pub fn from_env() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        match (env_nonempty("MEMVAULT_EMBED_URL"), env_nonempty("MEMVAULT_EMBED_MODEL")) {
            (None, None) => Ok(None),
            (Some(url), Some(model)) => Ok(Some(Embedder {
                client: reqwest::Client::new(),
                endpoint: format!("{}/embeddings", url.trim_end_matches('/')),
                model,
                api_key: env_nonempty("MEMVAULT_EMBED_API_KEY"),
            })),
            _ => Err("MEMVAULT_EMBED_URL and MEMVAULT_EMBED_MODEL must be set together".into()),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut request = self.client.post(&self.endpoint).json(&EmbedRequest { model: &self.model, input: text });
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|e| format!("{}: {e}", self.endpoint))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("{} returned {status}: {}", self.endpoint, body.trim()));
        }
        let parsed: EmbedResponse = response.json().await.map_err(|e| format!("{}: unexpected response shape: {e}", self.endpoint))?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("{} returned no embedding for model {:?}", self.endpoint, self.model))
    }
}
