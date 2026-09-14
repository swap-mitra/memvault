
use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::explain::{explain, search};
use crate::read_path::Query;
use crate::record::{NamespaceId, Outcome, SourceRef};
use crate::test_support::{fingerprint, harness, Harness};
use crate::write_path::{write_fact, WriteInput};


fn write(h: &mut Harness, content: &str, embedding: Vec<f32>, valid_from: chrono::DateTime<Utc>, valid_to: Option<chrono::DateTime<Utc>>) -> Uuid {
    write_ns(h, "default", content, embedding, valid_from, valid_to)
}

fn write_ns(
    h: &mut Harness,
    namespace: &str,
    content: &str,
    embedding: Vec<f32>,
    valid_from: chrono::DateTime<Utc>,
    valid_to: Option<chrono::DateTime<Utc>>,
) -> Uuid {
    write_fact(
        &h.ledger,
        &mut h.indexes,
        &mut h.keyring,
        WriteInput {
            namespace: NamespaceId(namespace.into()),
            content: content.as_bytes().to_vec(),
            embedding: Some(embedding),
            embedding_model: fingerprint(),
            valid_from,
            valid_to,
            fact_id: None,
            keywords: vec![],
            pinned: false,
            source: SourceRef::default(),
        },
    )
    .unwrap()
}

/// The plan's acceptance test: construct a query where at least one
/// candidate is cut by budget, one by k, one by the time filter; assert
/// all three appear in the Explanation list with correct outcome values,
/// not just the injected one.
#[test]
fn test_explanation_includes_all_outcomes() {
    let mut h = harness();
    let now = Utc::now();

    // Small content -> fits the tight token budget below.
    let injected = write(&mut h, "a", vec![1.0, 0.0, 0.0, 0.0], now - Duration::days(1), None);
    // Large content -> ranks 2nd (within k=2) but doesn't fit the budget.
    let big_content: String = "x".repeat(200);
    let cut_by_budget = write(&mut h, &big_content, vec![0.9, 0.436, 0.0, 0.0], now - Duration::days(1), None);
    // Ranks 3rd -- beyond k=2, so cut before budget packing is even reached.
    let cut_by_k = write(&mut h, "c", vec![0.5, 0.866, 0.0, 0.0], now - Duration::days(1), None);
    // Was valid, but its interval already closed before "now".
    let filtered_by_time = write(
        &mut h,
        "d",
        vec![0.0, 1.0, 0.0, 0.0],
        now - Duration::days(10),
        Some(now - Duration::days(1)),
    );

    h.indexes.keyword.commit().unwrap();

    let query = Query {
        text: None,
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        embedding_model: None,
        namespace: NamespaceId("default".into()),
        as_of: None,
        k: 2,
        max_tokens: 10, // enough for "injected"'s small ciphertext, not for the 200-byte one
    };

    let (explanations, retrieval_id) = search(&h.ledger, &h.indexes, query).unwrap();

    let outcome_of = |fact_id: Uuid| {
        explanations
            .iter()
            .find(|e| e.fact_id == fact_id)
            .unwrap_or_else(|| panic!("fact_id {fact_id} missing from explanations -- lossy output"))
            .outcome
    };

    assert_eq!(outcome_of(injected), Outcome::Injected);
    assert_eq!(outcome_of(cut_by_budget), Outcome::CutByBudget);
    assert_eq!(outcome_of(cut_by_k), Outcome::CutByK);
    assert_eq!(outcome_of(filtered_by_time), Outcome::FilteredByTime);
    assert_eq!(explanations.len(), 4, "every considered candidate must appear, not just the winners");

    // The full trail was also durably written as a Retrieval record.
    let retrieval_record = h
        .ledger
        .scan_from(0)
        .unwrap()
        .map(|r| r.unwrap())
        .find(|r| matches!(&r.payload, crate::record::Payload::Retrieval(ret) if ret.retrieval_id == retrieval_id))
        .expect("Retrieval record not found in ledger");
    match retrieval_record.payload {
        crate::record::Payload::Retrieval(ret) => assert_eq!(ret.candidates.len(), 4),
        _ => unreachable!(),
    }
}

#[test]
fn explain_reconstructs_a_past_retrieval_exactly() {
    let mut h = harness();
    write(&mut h, "hello", vec![1.0, 0.0, 0.0, 0.0], Utc::now(), None);
    h.indexes.keyword.commit().unwrap();

    let query = Query {
        text: Some("hello".into()),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        embedding_model: None,
        namespace: NamespaceId("default".into()),
        as_of: None,
        k: 5,
        max_tokens: 4096,
    };
    let (original, retrieval_id) = search(&h.ledger, &h.indexes, query).unwrap();

    let reconstructed = explain(&h.ledger, retrieval_id).unwrap();
    assert_eq!(reconstructed, original);
}

#[test]
fn explain_unknown_retrieval_id_is_not_found() {
    let h = harness();
    let result = explain(&h.ledger, Uuid::new_v4());
    assert!(matches!(result, Err(crate::explain::ExplainError::NotFound)));
}

/// The indexes are shared across every namespace in a data directory, so
/// the read path has to enforce the boundary itself -- nothing upstream of
/// it does.
///
/// Both facts here are byte-identical and share an embedding, so both are
/// certain to reach fusion. Only the querying namespace's own may survive,
/// and the other must not appear even as a cut candidate: the explanation
/// list is written verbatim into a `Retrieval` ledger record, so naming a
/// foreign fact there would persist its id and ledger position inside the
/// boundary this filter exists to hold.
#[test]
fn test_search_never_returns_another_namespaces_facts() {
    let mut h = harness();
    let now = Utc::now();

    let mine = write_ns(&mut h, "tenant-a", "shared secret alpha", vec![1.0, 0.0, 0.0, 0.0], now - Duration::days(1), None);
    let theirs = write_ns(&mut h, "tenant-b", "shared secret alpha", vec![1.0, 0.0, 0.0, 0.0], now - Duration::days(1), None);

    h.indexes.keyword.commit().unwrap();

    let (explanations, _) = search(
        &h.ledger,
        &h.indexes,
        Query {
            text: Some("shared secret alpha".into()),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            embedding_model: None,
            namespace: NamespaceId("tenant-a".into()),
            as_of: None,
            k: 10,
            max_tokens: 4096,
        },
    )
    .unwrap();

    let returned: Vec<Uuid> = explanations.iter().map(|e| e.fact_id).collect();
    assert!(returned.contains(&mine), "tenant-a's own fact is missing: {returned:?}");
    assert!(!returned.contains(&theirs), "tenant-b's fact leaked into tenant-a's search: {returned:?}");
    assert_eq!(returned.len(), 1, "only tenant-a's fact should have been considered at all: {returned:?}");
}

/// What the agent reads: the plaintext of every `Injected` candidate, in
/// packing order, and nothing for the ones that were cut.
#[test]
fn injected_contents_returns_only_what_was_injected_in_order() {
    let mut h = harness();
    let now = Utc::now();

    let best = write(&mut h, "the deploy script lives in ops/deploy.sh", vec![1.0, 0.0, 0.0, 0.0], now - Duration::days(1), None);
    let second = write(&mut h, "staging runs postgres 16", vec![0.9, 0.436, 0.0, 0.0], now - Duration::days(1), None);
    let big_content: String = "x".repeat(400);
    let cut = write(&mut h, &big_content, vec![0.8, 0.6, 0.0, 0.0], now - Duration::days(1), None);
    h.indexes.keyword.commit().unwrap();

    let (explanations, _) = search(
        &h.ledger,
        &h.indexes,
        Query {
            text: None,
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            embedding_model: None,
            namespace: NamespaceId("default".into()),
            as_of: None,
            k: 10,
            max_tokens: 40,
        },
    )
    .unwrap();

    let injected = crate::explain::injected_contents(&h.ledger, &h.keyring, &explanations).unwrap();
    let ids: Vec<Uuid> = injected.iter().map(|f| f.fact_id).collect();
    assert_eq!(ids, vec![best, second], "packing order, best score first: {ids:?}");
    assert!(!ids.contains(&cut), "a CutByBudget fact must not be handed back as content");
    assert_eq!(injected[0].content, b"the deploy script lives in ops/deploy.sh");
    assert_eq!(injected[1].content, b"staging runs postgres 16");
}
