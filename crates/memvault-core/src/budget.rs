//! Budget packing (product doc §6.4 step 6): greedy by `final_score`
//! descending, admitting whatever fits, skipping (not stopping at) an
//! oversized candidate so a smaller later one still gets a chance.
//! Optimal packing is knapsack; the doc is explicit that relevance
//! ordering beats a marginally tighter pack.

use uuid::Uuid;

use crate::decay::ScoredCandidate;

#[derive(Debug, Clone, PartialEq)]
pub struct PackedCandidate {
    pub candidate: ScoredCandidate,
    pub token_cost: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkippedCandidate {
    pub candidate: ScoredCandidate,
    pub token_cost: u32,
}

pub fn pack_to_budget(
    mut candidates: Vec<ScoredCandidate>,
    max_tokens: u32,
    token_cost: impl Fn(Uuid) -> u32,
) -> (Vec<PackedCandidate>, Vec<SkippedCandidate>) {
    candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).expect("final_score is never NaN"));

    let mut remaining = max_tokens;
    let mut packed = Vec::new();
    let mut skipped = Vec::new();

    for candidate in candidates {
        let cost = token_cost(candidate.fact_id);
        if cost <= remaining {
            remaining -= cost;
            packed.push(PackedCandidate { candidate, token_cost: cost });
        } else {
            skipped.push(SkippedCandidate { candidate, token_cost: cost });
        }
    }

    (packed, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(fact_id: Uuid, final_score: f32) -> ScoredCandidate {
        ScoredCandidate {
            fact_id,
            ann_rank: None,
            ann_distance: None,
            bm25_rank: None,
            bm25_score: None,
            rrf_score: final_score,
            decay_weight: 1.0,
            final_score,
        }
    }

    /// Acceptance test: an oversized candidate is skipped, not a stopping
    /// point -- a smaller, lower-ranked candidate after it still fits.
    #[test]
    fn test_budget_packing_skips_oversized_not_terminates() {
        let big = Uuid::from_u128(1);
        let small = Uuid::from_u128(2);
        let candidates = vec![scored(big, 0.9), scored(small, 0.5)];

        let costs = |fact_id: Uuid| if fact_id == big { 100 } else { 10 };
        let (packed, skipped) = pack_to_budget(candidates, 50, costs);

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].candidate.fact_id, small);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].candidate.fact_id, big);
    }

    #[test]
    fn packs_in_descending_score_order_when_everything_fits() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let candidates = vec![scored(a, 0.1), scored(b, 0.9)];

        let (packed, skipped) = pack_to_budget(candidates, 1000, |_| 10);

        assert!(skipped.is_empty());
        assert_eq!(packed[0].candidate.fact_id, b);
        assert_eq!(packed[1].candidate.fact_id, a);
    }
}
