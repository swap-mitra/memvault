//! Vector and keyword indexes (product doc §5, §6.4). Both are derived
//! state: rebuildable from the ledger at any time, never a source of
//! truth. They share one error type since callers (the read/write paths)
//! treat both backends the same way -- an index operation either works or
//! it doesn't, and the caller doesn't care which storage engine failed.

pub mod keyword;
pub mod vector;

#[cfg(test)]
mod keyword_tests;
#[cfg(test)]
mod vector_tests;

pub use keyword::KeywordIndex;
pub use vector::VectorIndex;

/// The two derived, rebuildable-from-the-ledger indexes bundled together,
/// since every caller of the read/write paths needs both at once.
pub struct Indexes {
    pub vector: VectorIndex,
    pub keyword: KeywordIndex,
}

#[derive(Debug)]
pub enum IndexError {
    Usearch(cxx::Exception),
    Redb(redb::Error),
    Io(std::io::Error),
    /// tantivy::TantivyError and tantivy::query::QueryParserError, flattened
    /// to a message: both are tantivy-originated and neither needs to be
    /// matched on by callers here, just reported.
    Tantivy(String),
    Corrupt(String),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Usearch(e) => write!(f, "usearch error: {e}"),
            IndexError::Redb(e) => write!(f, "index sidecar storage error: {e}"),
            IndexError::Io(e) => write!(f, "index sidecar io error: {e}"),
            IndexError::Tantivy(e) => write!(f, "tantivy error: {e}"),
            IndexError::Corrupt(msg) => write!(f, "index sidecar corrupt: {msg}"),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<cxx::Exception> for IndexError {
    fn from(e: cxx::Exception) -> Self {
        IndexError::Usearch(e)
    }
}

impl From<std::io::Error> for IndexError {
    fn from(e: std::io::Error) -> Self {
        IndexError::Io(e)
    }
}

macro_rules! redb_error {
    ($t:ty) => {
        impl From<$t> for IndexError {
            fn from(e: $t) -> Self {
                IndexError::Redb(e.into())
            }
        }
    };
}

redb_error!(redb::DatabaseError);
redb_error!(redb::TransactionError);
redb_error!(redb::TableError);
redb_error!(redb::StorageError);
redb_error!(redb::CommitError);
