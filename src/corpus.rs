use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use crate::{BenchmarkConfig, Error, Result};

#[derive(Debug, Clone)]
pub struct Corpus {
    text: String,
    char_offsets: Vec<usize>,
}

impl Corpus {
    pub async fn load(config: &BenchmarkConfig, client: &reqwest::Client) -> Result<Self> {
        let text = if let Some(path) = &config.prompt_file {
            tokio::fs::read_to_string(path)
                .await
                .map_err(|source| Error::ReadFile {
                    path: path.clone(),
                    source,
                })?
        } else {
            load_url(&config.book_url, client).await?
        };

        let text = strip_gutenberg_header(text);
        let char_offsets = text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(text.len()))
            .collect::<Vec<_>>();
        if char_offsets.len() < 2 {
            return Err(Error::Config("the prompt corpus is empty".into()));
        }
        Ok(Self { text, char_offsets })
    }

    pub fn len_chars(&self) -> usize {
        self.char_offsets.len() - 1
    }

    pub fn slice_chars(&self, start: usize, len: usize) -> String {
        let total = self.len_chars();
        if len == 0 || total == 0 {
            return String::new();
        }
        if len <= total && start + len <= total {
            return self.text[self.char_offsets[start]..self.char_offsets[start + len]].to_owned();
        }

        let mut result = String::with_capacity(len.saturating_mul(2));
        let mut remaining = len;
        let mut cursor = start % total;
        while remaining > 0 {
            let take = remaining.min(total - cursor);
            result
                .push_str(&self.text[self.char_offsets[cursor]..self.char_offsets[cursor + take]]);
            remaining -= take;
            cursor = 0;
        }
        result
    }

    pub fn calibration_sample(&self, chars: usize) -> String {
        self.slice_chars(0, chars.max(1))
    }
}

async fn load_url(url: &str, client: &reqwest::Client) -> Result<String> {
    let cache_file = cache_path(url)?;
    if let Ok(text) = tokio::fs::read_to_string(&cache_file).await {
        eprintln!("Loading corpus from {}", cache_file.display());
        return Ok(text);
    }

    eprintln!("Downloading corpus from {url} ...");
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|source| Error::Request {
            url: url.to_owned(),
            source,
        })?;
    let status = response.status();
    let body = response.text().await.map_err(|source| Error::Request {
        url: url.to_owned(),
        source,
    })?;
    if !status.is_success() {
        return Err(Error::HttpStatus {
            url: url.to_owned(),
            status,
            body,
        });
    }
    if let Some(parent) = cache_file.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| Error::WriteFile {
                path: parent.to_owned(),
                source,
            })?;
    }
    tokio::fs::write(&cache_file, body.as_bytes())
        .await
        .map_err(|source| Error::WriteFile {
            path: cache_file.clone(),
            source,
        })?;
    Ok(body)
}

fn cache_path(url: &str) -> Result<PathBuf> {
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    let root = dirs::cache_dir()
        .ok_or_else(|| Error::Config("could not determine the user cache directory".into()))?;
    Ok(root
        .join("rust-benchy")
        .join(format!("{:016x}.txt", hasher.finish())))
}

fn strip_gutenberg_header(text: String) -> String {
    const MARKER: &str = "*** START OF THE PROJECT GUTENBERG EBOOK";
    match text.find(MARKER) {
        Some(index) => text[index..].to_owned(),
        None => text,
    }
}

#[allow(dead_code)]
fn _is_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_unicode_by_characters_and_wraps() {
        let text = "aé日".to_owned();
        let corpus = Corpus {
            char_offsets: text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(text.len()))
                .collect(),
            text,
        };
        assert_eq!(corpus.slice_chars(1, 2), "é日");
        assert_eq!(corpus.slice_chars(2, 4), "日aé日");
    }
}
