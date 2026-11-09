//! Caching PROJ objects, which cost more to build than a morsel costs to process.

use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use polars::prelude::PolarsResult;

/// The most values a cache holds.
pub(crate) const CACHE_CAPACITY: usize = 16;

// TODO: use pl hashmap, is faster.
// TODO: make a second 'cache' which literally stores the pointer of the key,
// so that it can re-use very quickly for the last used CRS (same loc in memory from header)
/// Values that are costly to build, by key, for a `thread_local!`.
/// A full cache is emptied before the next value goes in.
pub(crate) struct ThreadCache<K, V>(RefCell<HashMap<K, Rc<V>>>);

impl<K, V> Default for ThreadCache<K, V> {
    fn default() -> Self {
        Self(RefCell::new(HashMap::new()))
    }
}

impl<K: Hash + Eq, V> ThreadCache<K, V> {
    /// The value for `key`, built by `build` if it is not cached yet.
    /// A value that fails to build is not cached, and is built again the next time.
    pub(crate) fn get_or_build<Q>(
        &self,
        key: &Q,
        build: impl FnOnce() -> PolarsResult<V>,
    ) -> PolarsResult<Rc<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
    {
        if let Some(value) = self.0.borrow().get(key) {
            return Ok(Rc::clone(value));
        }
        // Built outside the borrow, so nothing is held while PROJ runs.
        let value = Rc::new(build()?);
        let mut values = self.0.borrow_mut();
        if values.len() >= CACHE_CAPACITY {
            values.clear();
        }
        values.insert(key.to_owned(), Rc::clone(&value));
        Ok(value)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.borrow().len()
    }
}
