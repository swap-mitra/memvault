//! `memvault.toml`, read from the data directory when the stores open.
//!
//! Product doc §6.9 describes per-namespace decay configuration; until now
//! every knob was a constant and the `ponytail:` notes said "once a config
//! loader exists". This is that loader. The file is optional and so is
//! every key in it; `memvault config` prints the effective values, which
//! doubles as the documentation of what can be set.
//!
//! ```toml
//! [decay]
//! half_life_days = 30.0
//! floor = 0.15
//!
//! [limits]
//! max_content_bytes = 65536
//!
//! [retrievals]
//! keep_days = 90            # unset: keep every retrieval forever
//!
//! [namespaces.project]      # per-namespace decay, unset fields fall back to [decay]
//! half_life_days = 7.0
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::decay::DecayConfig;
use crate::record::NamespaceId;
use crate::write_path::DEFAULT_MAX_CONTENT_BYTES;

pub const CONFIG_FILE: &str = "memvault.toml";

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub decay: DecaySection,
    pub limits: Limits,
    pub retrievals: Retention,
    /// Per-namespace decay overrides keyed by namespace name. A field left
    /// unset falls back to `[decay]`.
    pub namespaces: BTreeMap<String, DecayOverride>,
}

/// The decay prior every namespace gets unless it overrides it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecaySection {
    pub half_life_days: f64,
    pub floor: f64,
}

impl Default for DecaySection {
    fn default() -> Self {
        let d = DecayConfig::default();
        DecaySection { half_life_days: d.half_life_days, floor: d.floor }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// Largest fact a write accepts, in bytes of content.
    pub max_content_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_content_bytes: DEFAULT_MAX_CONTENT_BYTES }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Retention {
    /// Retrieval records older than this many days are pruned from the
    /// front of the retrievals chain when the server starts, and by
    /// `memvault prune`. Unset keeps every retrieval forever.
    pub keep_days: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecayOverride {
    pub half_life_days: Option<f64>,
    pub floor: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{} is not valid TOML for MemVault: {source}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("{}: {reason}", path.display())]
    Invalid { path: PathBuf, reason: String },
}

impl Config {
    /// The directory's configuration: `memvault.toml` if present, the
    /// defaults if not. A file that is present but wrong is an error, not
    /// a fallback -- a typo in a decay setting should not silently rank
    /// memories by the defaults.
    pub fn load(data_dir: &Path) -> Result<Config, ConfigError> {
        let path = data_dir.join(CONFIG_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
            Err(source) => return Err(ConfigError::Io { path, source }),
        };
        let config: Config = toml::from_str(&text).map_err(|source| ConfigError::Parse { path: path.clone(), source })?;
        config.validate().map_err(|reason| ConfigError::Invalid { path, reason })?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        check_decay("[decay]", self.decay.half_life_days, self.decay.floor)?;
        for name in self.namespaces.keys() {
            let d = self.decay_for(&NamespaceId(name.clone()));
            check_decay(&format!("[namespaces.{name}]"), d.half_life_days, d.floor)?;
        }
        if self.limits.max_content_bytes == 0 {
            return Err("[limits] max_content_bytes must be at least 1".into());
        }
        Ok(())
    }

    /// The decay prior for one namespace: its override where set, `[decay]`
    /// otherwise.
    pub fn decay_for(&self, namespace: &NamespaceId) -> DecayConfig {
        let over = self.namespaces.get(&namespace.0);
        DecayConfig {
            half_life_days: over.and_then(|o| o.half_life_days).unwrap_or(self.decay.half_life_days),
            floor: over.and_then(|o| o.floor).unwrap_or(self.decay.floor),
        }
    }

    /// The effective configuration as TOML, defaults filled in: what
    /// `memvault config` prints.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("Config always serialises")
    }
}

fn check_decay(section: &str, half_life_days: f64, floor: f64) -> Result<(), String> {
    if !(half_life_days > 0.0) {
        return Err(format!("{section} half_life_days must be greater than 0, got {half_life_days}"));
    }
    if !(0.0..=1.0).contains(&floor) {
        return Err(format!("{section} floor must be between 0 and 1, got {floor}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::tmp;

    fn load_from(text: &str) -> Result<Config, ConfigError> {
        let dir = tmp();
        std::fs::write(dir.path().join(CONFIG_FILE), text).unwrap();
        Config::load(dir.path())
    }

    #[test]
    fn a_missing_file_means_the_defaults() {
        let dir = tmp();
        let config = Config::load(dir.path()).unwrap();
        assert_eq!(config, Config::default());
        assert_eq!(config.decay_for(&NamespaceId("anything".into())), DecayConfig::default());
        assert_eq!(config.limits.max_content_bytes, DEFAULT_MAX_CONTENT_BYTES);
        assert_eq!(config.retrievals.keep_days, None);
    }

    #[test]
    fn namespaces_override_only_what_they_set() {
        let config = load_from(
            r#"
            [decay]
            half_life_days = 10.0
            floor = 0.2

            [retrievals]
            keep_days = 30

            [namespaces.project]
            half_life_days = 3.0
            "#,
        )
        .unwrap();
        let project = config.decay_for(&NamespaceId("project".into()));
        assert_eq!(project.half_life_days, 3.0);
        assert_eq!(project.floor, 0.2, "floor falls back to [decay]");
        assert_eq!(config.decay_for(&NamespaceId("other".into())).half_life_days, 10.0);
        assert_eq!(config.retrievals.keep_days, Some(30));
    }

    #[test]
    fn bad_values_and_unknown_keys_are_errors_not_defaults() {
        let err = load_from("[decay]\nfloor = 1.5\n").unwrap_err();
        assert!(matches!(err, ConfigError::Invalid { .. }), "{err}");

        let err = load_from("[namespaces.p]\nhalf_life_days = 0.0\n").unwrap_err();
        assert!(err.to_string().contains("[namespaces.p]"), "{err}");

        let err = load_from("[decay]\nhalf_life = 30\n").unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }), "a misspelt key must not be ignored: {err}");
    }

    #[test]
    fn printed_config_reloads_to_itself() {
        let config = load_from("[limits]\nmax_content_bytes = 1024\n[namespaces.a]\nfloor = 0.5\n").unwrap();
        let reloaded: Config = toml::from_str(&config.to_toml()).unwrap();
        assert_eq!(reloaded, config);
    }
}
