//! Decay weighting (product doc §6.4 step 5, §3 P3): a ranking prior, never
//! a truth model. Pinned facts bypass it entirely.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::read_path::FusedCandidate;

pub struct DecayConfig {
    pub half_life_days: f64,
    pub floor: f64,
}

/// `w(f) = max(floor, exp(-ln(2) * age_days / half_life_days))`.
pub fn decay_weight(age_days: f64, half_life_days: f64, floor: f64) -> f64 {
    (-std::f64::consts::LN_2 * age_days / half_life_days).exp().max(floor)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub fact_id: Uuid,
    pub ann_rank: Option<u32>,
    pub ann_distance: Option<f32>,
    pub bm25_rank: Option<u32>,
    pub bm25_score: Option<f32>,
    pub rrf_score: f32,
    pub decay_weight: f32,
    pub final_score: f32,
}

/// `final(f) = rrf(f) * (pinned(f) ? 1.0 : w(f))`. `age_days` is measured
/// from `last_accessed(fact_id)`, not the fact's creation time, so a
/// retrieval reinforces it (product doc §6.4) -- callers inject
/// `last_accessed` rather than this module owning where that state lives,
/// since no such store exists yet.
pub fn apply_decay(
    candidates: Vec<FusedCandidate>,
    pinned: impl Fn(Uuid) -> bool,
    last_accessed: impl Fn(Uuid) -> DateTime<Utc>,
    now: DateTime<Utc>,
    cfg: &DecayConfig,
) -> Vec<ScoredCandidate> {
    candidates
        .into_iter()
        .map(|c| {
            let weight = if pinned(c.fact_id) {
                1.0
            } else {
                let age_days = (now - last_accessed(c.fact_id)).num_seconds() as f64 / 86_400.0;
                decay_weight(age_days, cfg.half_life_days, cfg.floor)
            };
            let final_score = c.rrf_score * weight as f32;
            ScoredCandidate {
                fact_id: c.fact_id,
                ann_rank: c.ann_rank,
                ann_distance: c.ann_distance,
                bm25_rank: c.bm25_rank,
                bm25_score: c.bm25_score,
                rrf_score: c.rrf_score,
                decay_weight: weight as f32,
                final_score,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(fact_id: Uuid, rrf_score: f32) -> FusedCandidate {
        FusedCandidate {
            fact_id,
            ann_rank: Some(0),
            ann_distance: Some(0.1),
            bm25_rank: None,
            bm25_score: None,
            rrf_score,
        }
    }

    #[test]
    fn zero_age_gives_full_weight() {
        assert_eq!(decay_weight(0.0, 30.0, 0.15), 1.0);
    }

    #[test]
    fn one_half_life_halves_weight() {
        let w = decay_weight(30.0, 30.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn floor_bounds_unlimited_decay() {
        let w = decay_weight(10_000.0, 30.0, 0.15);
        assert_eq!(w, 0.15);
    }

    /// Acceptance test: pinned facts bypass decay entirely, regardless of
    /// how stale `last_accessed` claims they are.
    #[test]
    fn test_pinned_fact_ignores_decay() {
        let fact_id = Uuid::from_u128(1);
        let now = DateTime::from_timestamp(2_000_000_000, 0).unwrap();
        let ancient = DateTime::from_timestamp(0, 0).unwrap();
        let cfg = DecayConfig { half_life_days: 30.0, floor: 0.0 };

        let results = apply_decay(vec![candidate(fact_id, 0.5)], |_| true, |_| ancient, now, &cfg);

        assert_eq!(results[0].decay_weight, 1.0);
        assert_eq!(results[0].final_score, 0.5);
    }

    #[test]
    fn age_is_measured_from_last_access_not_a_fixed_creation_time() {
        let fact_id = Uuid::from_u128(1);
        let now = DateTime::from_timestamp(1_000_000_000, 0).unwrap();
        let cfg = DecayConfig { half_life_days: 30.0, floor: 0.0 };

        // Same candidate, two different `last_accessed` values: a recent
        // access should score higher than a stale one, proving the age
        // comes from the injected accessor, not some fixed field.
        let recent = apply_decay(vec![candidate(fact_id, 1.0)], |_| false, |_| now, now, &cfg);
        let stale = apply_decay(
            vec![candidate(fact_id, 1.0)],
            |_| false,
            |_| now - chrono::Duration::days(90),
            now,
            &cfg,
        );

        assert_eq!(recent[0].decay_weight, 1.0);
        assert!(stale[0].decay_weight < recent[0].decay_weight);
    }
}
