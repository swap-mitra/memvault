//! Keyword index: a tantivy BM25 index over content and caller-supplied
//! keyword boost terms (product doc §6.1's `Assert.keywords`).
//!
//! Follows tantivy's own writer-commits/readers-reload model directly
//! (product doc §6.7) rather than layering bespoke locking on top: the
//! reader is `ReloadPolicy::Manual` and `commit()` reloads it explicitly,
//! so a caller who has called `commit()` is guaranteed to search the
//! version they just wrote.

use std::path::{Path, PathBuf};

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{document::Value, Field, Schema, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};
use uuid::Uuid;

use super::IndexError;

/// Keywords count several times over in the query's default fields so they
/// out-rank equal-frequency terms that only appear in free-text content --
/// a plain, if blunt, stand-in for a real per-field boost.
const KEYWORD_FIELD_BOOST: tantivy::Score = 3.0;

fn watermark_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".watermark");
    PathBuf::from(os)
}

fn read_watermark(path: &Path) -> Result<u64, IndexError> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| IndexError::Corrupt("watermark file has the wrong length".into()))?;
            Ok(u64::from_le_bytes(arr))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e.into()),
    }
}

pub struct KeywordIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    fact_id_field: Field,
    content_field: Field,
    keywords_field: Field,
    watermark_path: PathBuf,
    watermark: u64,
}

impl KeywordIndex {
    pub fn open_or_create(path: &Path) -> Result<Self, IndexError> {
        std::fs::create_dir_all(path)?;

        let mut schema_builder = Schema::builder();
        let fact_id_field = schema_builder.add_text_field("fact_id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let keywords_field = schema_builder.add_text_field("keywords", TEXT);
        let schema = schema_builder.build();

        let dir = tantivy::directory::MmapDirectory::open(path).map_err(|e| IndexError::Tantivy(e.to_string()))?;
        let index = Index::open_or_create(dir, schema).map_err(|e| IndexError::Tantivy(e.to_string()))?;

        let writer = index
            .writer::<TantivyDocument>(50_000_000)
            .map_err(|e| IndexError::Tantivy(e.to_string()))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e: tantivy::TantivyError| IndexError::Tantivy(e.to_string()))?;

        let watermark_path = watermark_path(path);
        let watermark = read_watermark(&watermark_path)?;

        Ok(KeywordIndex {
            index,
            writer,
            reader,
            fact_id_field,
            content_field,
            keywords_field,
            watermark_path,
            watermark,
        })
    }

    pub fn insert(&mut self, fact_id: Uuid, content: &str, keywords: &[String]) -> Result<(), IndexError> {
        let mut doc = TantivyDocument::default();
        doc.add_text(self.fact_id_field, fact_id.to_string());
        doc.add_text(self.content_field, content);
        for keyword in keywords {
            doc.add_text(self.keywords_field, keyword);
        }
        self.writer.add_document(doc).map_err(|e| IndexError::Tantivy(e.to_string()))?;
        Ok(())
    }

    pub fn remove(&mut self, fact_id: Uuid) -> Result<(), IndexError> {
        let term = Term::from_field_text(self.fact_id_field, &fact_id.to_string());
        self.writer.delete_term(term);
        Ok(())
    }

    /// Commits pending writes and reloads the reader, so a search
    /// immediately after this call sees them.
    pub fn commit(&mut self) -> Result<(), IndexError> {
        self.writer.commit().map_err(|e| IndexError::Tantivy(e.to_string()))?;
        self.reader.reload().map_err(|e| IndexError::Tantivy(e.to_string()))?;
        Ok(())
    }

    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(Uuid, f32, u32)>, IndexError> {
        let searcher = self.reader.searcher();
        let mut query_parser = QueryParser::for_index(&self.index, vec![self.content_field, self.keywords_field]);
        query_parser.set_field_boost(self.keywords_field, KEYWORD_FIELD_BOOST);
        let parsed = query_parser.parse_query(query).map_err(|e| IndexError::Tantivy(e.to_string()))?;

        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(k).order_by_score())
            .map_err(|e| IndexError::Tantivy(e.to_string()))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (rank, (score, doc_address)) in top_docs.into_iter().enumerate() {
            let doc: TantivyDocument = searcher.doc(doc_address).map_err(|e| IndexError::Tantivy(e.to_string()))?;
            let fact_id_str = doc
                .get_first(self.fact_id_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| IndexError::Corrupt("stored document missing fact_id field".into()))?;
            let fact_id = Uuid::parse_str(fact_id_str)
                .map_err(|_| IndexError::Corrupt("stored fact_id is not a valid uuid".into()))?;
            results.push((fact_id, score, rank as u32));
        }
        Ok(results)
    }

    pub fn watermark(&self) -> u64 {
        self.watermark
    }

    /// See `VectorIndex::set_watermark` -- same addition beyond the plan's
    /// interface sketch, needed for recovery to persist progress.
    pub fn set_watermark(&mut self, seq: u64) -> Result<(), IndexError> {
        self.watermark = seq;
        std::fs::write(&self.watermark_path, seq.to_le_bytes())?;
        Ok(())
    }

    /// Discards every document and resets the watermark to 0. Used by
    /// recovery when the watermark is in an impossible state and this
    /// index cannot be trusted incrementally.
    pub fn reset(&mut self) -> Result<(), IndexError> {
        self.writer.delete_all_documents().map_err(|e| IndexError::Tantivy(e.to_string()))?;
        self.commit()?;
        self.set_watermark(0)
    }
}
