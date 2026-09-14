//! The retrievals chain: its own file, an index by retrieval_id, and
//! pruning from the front that leaves the remainder verifying.

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::ledger::{Ledger, RETRIEVALS_FILE};
use crate::record::{NamespaceId, Payload, Retrieval};
use crate::test_support::tmp;

fn retrieval(id: Uuid) -> Payload {
    Payload::Retrieval(Retrieval {
        retrieval_id: id,
        query_text: Some("deploy script".into()),
        query_embedding_model: None,
        as_of: None,
        max_tokens: 2048,
        k: 10,
        candidates: vec![],
    })
}

fn namespace() -> NamespaceId {
    NamespaceId("default".into())
}

#[test]
fn retrievals_have_their_own_chain_and_an_index() {
    let dir = tmp();
    let ledger = Ledger::open(&dir.path().join("ledger.redb")).unwrap();
    let id = Uuid::new_v4();

    let seq = ledger.append(namespace(), Utc::now(), retrieval(id)).unwrap();

    assert_eq!(seq, 0);
    assert_eq!(ledger.head().unwrap(), 0, "the facts chain is untouched by a search");
    assert_eq!(ledger.retrievals_head().unwrap(), 1);
    assert!(dir.path().join(RETRIEVALS_FILE).exists());

    let found = ledger.find_retrieval(id).unwrap().expect("indexed on write");
    assert!(matches!(found.payload, Payload::Retrieval(r) if r.retrieval_id == id));
    assert!(ledger.find_retrieval(Uuid::new_v4()).unwrap().is_none());

    ledger.verify().unwrap();
    ledger.verify_retrievals().unwrap();
}

#[test]
fn pruning_drops_old_retrievals_and_the_rest_still_verifies() {
    let dir = tmp();
    let path = dir.path().join("ledger.redb");
    let ledger = Ledger::open(&path).unwrap();
    let base = Utc::now() - Duration::days(10);
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    for (day, id) in ids.iter().enumerate() {
        ledger.append(namespace(), base + Duration::days(day as i64), retrieval(*id)).unwrap();
    }

    // Days 0, 1 and 2 fall before a cutoff at two and a half days in.
    let outcome = ledger.prune_retrievals_before(base + Duration::hours(60)).unwrap();
    assert_eq!(outcome.pruned, 3);
    assert_eq!(outcome.first_seq, Some(3));

    ledger.verify_retrievals().unwrap();
    assert!(ledger.find_retrieval(ids[0]).unwrap().is_none(), "pruned: no longer explainable");
    assert!(ledger.find_retrieval(ids[3]).unwrap().is_some());

    // Seqs keep counting from the retained anchor; nothing is renumbered.
    let next = ledger.append(namespace(), Utc::now(), retrieval(Uuid::new_v4())).unwrap();
    assert_eq!(next, 5);
    assert_eq!(ledger.retrievals_head().unwrap(), 6);
    ledger.verify_retrievals().unwrap();

    // Survives a reopen: the pruned chain is what is on disk.
    drop(ledger);
    let reopened = Ledger::open(&path).unwrap();
    assert_eq!(reopened.retrievals_first_seq().unwrap(), Some(3));
    reopened.verify_retrievals().unwrap();
    assert!(reopened.find_retrieval(ids[4]).unwrap().is_some());
}

#[test]
fn pruning_never_removes_the_newest_record() {
    let dir = tmp();
    let ledger = Ledger::open(&dir.path().join("ledger.redb")).unwrap();
    let old = Utc::now() - Duration::days(400);
    for _ in 0..3 {
        ledger.append(namespace(), old, retrieval(Uuid::new_v4())).unwrap();
    }

    // Everything is older than the cutoff; the anchor stays regardless.
    let outcome = ledger.prune_retrievals_before(Utc::now()).unwrap();
    assert_eq!(outcome.pruned, 2);
    assert_eq!(outcome.first_seq, Some(2));
    ledger.verify_retrievals().unwrap();

    // Nothing left to prune, and asking again is harmless.
    let again = ledger.prune_retrievals_before(Utc::now()).unwrap();
    assert_eq!(again.pruned, 0);
    assert_eq!(again.first_seq, Some(2));
}
