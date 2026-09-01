use rand::Rng;
use uuid::Uuid;

use crate::corpus::Corpus;

#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    pub user_chars_per_token: f64,
    pub context_chars_per_token: f64,
    pub user_overhead_tokens: f64,
    pub context_overhead_tokens: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            user_chars_per_token: 4.0,
            context_chars_per_token: 4.0,
            user_overhead_tokens: 0.0,
            context_overhead_tokens: 0.0,
        }
    }
}

impl Calibration {
    pub fn from_samples(
        short_chars: usize,
        long_chars: usize,
        user_short_tokens: usize,
        user_long_tokens: usize,
        context_short_tokens: usize,
        context_long_tokens: usize,
    ) -> Self {
        fn fit(
            short_chars: usize,
            long_chars: usize,
            short_tokens: usize,
            long_tokens: usize,
        ) -> (f64, f64) {
            let token_delta = long_tokens.saturating_sub(short_tokens);
            if token_delta == 0 || long_chars <= short_chars {
                return (4.0, 0.0);
            }
            let chars_per_token = (long_chars - short_chars) as f64 / token_delta as f64;
            let overhead = short_tokens as f64 - short_chars as f64 / chars_per_token;
            (chars_per_token.clamp(0.5, 20.0), overhead.max(0.0))
        }
        let (user_chars_per_token, user_overhead_tokens) =
            fit(short_chars, long_chars, user_short_tokens, user_long_tokens);
        let (context_chars_per_token, context_overhead_tokens) = fit(
            short_chars,
            long_chars,
            context_short_tokens,
            context_long_tokens,
        );
        Self {
            user_chars_per_token,
            context_chars_per_token,
            user_overhead_tokens,
            context_overhead_tokens,
        }
    }

    fn user_chars(&self, tokens: usize, adapt: bool) -> usize {
        let overhead = if adapt {
            self.user_overhead_tokens
        } else {
            0.0
        };
        ((tokens as f64 - overhead).max(1.0) * self.user_chars_per_token).round() as usize
    }

    fn context_chars(&self, tokens: usize, adapt: bool) -> usize {
        let overhead = if adapt {
            self.context_overhead_tokens
        } else {
            0.0
        };
        ((tokens as f64 - overhead).max(1.0) * self.context_chars_per_token).round() as usize
    }
}

#[derive(Clone)]
pub struct PromptGenerator {
    corpus: Corpus,
    calibration: Calibration,
    adapt_overhead: bool,
}

impl PromptGenerator {
    pub fn new(corpus: Corpus, calibration: Calibration, adapt_overhead: bool) -> Self {
        Self {
            corpus,
            calibration,
            adapt_overhead,
        }
    }

    pub fn generate(
        &self,
        prompt_tokens: usize,
        context_tokens: usize,
        no_cache: bool,
    ) -> (String, String) {
        let context_chars = if context_tokens == 0 {
            0
        } else {
            self.calibration
                .context_chars(context_tokens, self.adapt_overhead)
        };
        let suffix = if no_cache {
            format!(" {}", Uuid::new_v4())
        } else {
            String::new()
        };
        let suffix_estimate = if suffix.is_empty() {
            0
        } else {
            suffix.chars().count()
        };
        let prompt_chars = self
            .calibration
            .user_chars(prompt_tokens, self.adapt_overhead)
            .saturating_sub(suffix_estimate);
        let needed = context_chars.saturating_add(prompt_chars);
        let available = self.corpus.len_chars();
        let max_start = available.saturating_sub(needed.min(available));
        let start = if max_start == 0 {
            0
        } else {
            rand::rng().random_range(0..=max_start)
        };
        let context = self.corpus.slice_chars(start, context_chars);
        let mut prompt = self.corpus.slice_chars(start + context_chars, prompt_chars);
        prompt.push_str(&suffix);
        (context, prompt)
    }

    pub fn generate_batch(
        &self,
        count: usize,
        pp: usize,
        depth: usize,
        no_cache: bool,
    ) -> Vec<(String, String)> {
        (0..count)
            .map(|_| self.generate(pp, depth, no_cache))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_fits_linear_token_counts() {
        let c = Calibration::from_samples(100, 1000, 30, 255, 40, 265);
        assert!((c.user_chars_per_token - 4.0).abs() < 0.001);
        assert!((c.user_overhead_tokens - 5.0).abs() < 0.001);
        assert!((c.context_overhead_tokens - 15.0).abs() < 0.001);
    }
}
