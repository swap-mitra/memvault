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

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("usearch error: {0}")]
    Usearch(#[from] cxx::Exception),
    #[error("index sidecar storage error: {0}")]
    Redb(redb::Error),
    #[error("index sidecar io error: {0}")]
    Io(#[from] std::io::Error),
    /// tantivy::TantivyError and tantivy::query::QueryParserError, flattened
    /// to a message: both are tantivy-originated and neither needs to be
    /// matched on by callers here, just reported.
    #[error("tantivy error: {0}")]
    Tantivy(String),
    #[error("index sidecar corrupt: {0}")]
    Corrupt(String),
}

impl From<tantivy::TantivyError> for IndexError {
    fn from(e: tantivy::TantivyError) -> Self {
        IndexError::Tantivy(e.to_string())
    }
}

impl From<tantivy::query::QueryParserError> for IndexError {
    fn from(e: tantivy::query::QueryParserError) -> Self {
        IndexError::Tantivy(e.to_string())
    }
}

impl From<tantivy::directory::error::OpenDirectoryError> for IndexError {
    fn from(e: tantivy::directory::error::OpenDirectoryError) -> Self {
        IndexError::Tantivy(e.to_string())
    }
}

crate::redb_error!(IndexError, IndexError::Redb);
