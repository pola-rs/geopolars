//! Caching PROJ objects, which cost more to build than a morsel costs to process.

use std::borrow::Borrow;
use std::cell::RefCell;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::{LazyLock, Mutex, PoisonError};

use polars::prelude::{PlHashMap, PolarsResult};

/// The most values a cache holds.
pub(crate) const CACHE_CAPACITY: usize = 16;

/// Values that are costly to build, by key, for a `thread_local!`.
/// A full cache is emptied before the next value goes in.
pub(crate) struct ThreadCache<K, V>(RefCell<Values<K, V>>);

struct Values<K, V> {
    all: PlHashMap<K, Rc<V>>,
    /// Compared by content, not by address.
    /// Each plugin call imports its dtype afresh, into a new `Arc` of metadata,
    /// so the same CRS is never at the same address twice.
    last: Option<(K, Rc<V>)>,
}

impl<K, V> Default for ThreadCache<K, V> {
    fn default() -> Self {
        Self(RefCell::new(Values {
            all: PlHashMap::default(),
            last: None,
        }))
    }
}

impl<K: Hash + Eq + Clone, V> ThreadCache<K, V> {
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
        let mut values = self.0.borrow_mut();
        if let Some((last_key, value)) = &values.last {
            if last_key.borrow() == key {
                return Ok(Rc::clone(value));
            }
        }
        if let Some((cached_key, value)) = values.all.get_key_value(key) {
            let (cached_key, value) = (cached_key.clone(), Rc::clone(value));
            values.last = Some((cached_key, Rc::clone(&value)));
            return Ok(value);
        }
        // Built outside the borrow, so nothing is held while PROJ runs.
        drop(values);
        let value = Rc::new(build()?);
        let mut values = self.0.borrow_mut();
        let key = key.to_owned();
        insert_emptying_when_full(&mut values.all, key.clone(), Rc::clone(&value));
        values.last = Some((key, Rc::clone(&value)));
        Ok(value)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.borrow().all.len()
    }
}

/// Values that are costly to build, by key, for a `static` that every thread uses.
/// Like [ThreadCache], a full cache is emptied before the next value goes in,
/// and a value that fails to build is not cached.
pub(crate) struct SharedCache<K, V>(LazyLock<Mutex<PlHashMap<K, V>>>);

impl<K, V> SharedCache<K, V> {
    pub(crate) const fn new() -> Self {
        Self(LazyLock::new(Default::default))
    }
}

impl<K: Hash + Eq, V> SharedCache<K, V> {
    /// Applies `use_value` to the value for `key`, built by `build` if it is not cached yet.
    /// The lock is held for both, so a thread waits for a value another one is building,
    /// and only one thread at a time uses a value.
    pub(crate) fn with_or_build<Q, R>(
        &self,
        key: &Q,
        build: impl FnOnce() -> PolarsResult<V>,
        use_value: impl FnOnce(&V) -> PolarsResult<R>,
    ) -> PolarsResult<R>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
    {
        let mut values = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if !values.contains_key(key) {
            insert_emptying_when_full(&mut values, key.to_owned(), build()?);
        }
        use_value(&values[key])
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

fn insert_emptying_when_full<K: Hash + Eq, V>(values: &mut PlHashMap<K, V>, key: K, value: V) {
    if values.len() >= CACHE_CAPACITY {
        values.clear();
    }
    values.insert(key, value);
}

#[cfg(test)]
mod tests {
    //! The caching itself cannot be seen from Python.

    use std::cell::Cell;

    use super::*;

    #[test]
    fn builds_each_key_once_and_serves_the_same_value() {
        let cache = ThreadCache::<String, usize>::default();
        let builds = Cell::new(0);
        let get = |key: &str| {
            cache
                .get_or_build(key, || {
                    builds.set(builds.get() + 1);
                    Ok(key.len())
                })
                .unwrap()
        };

        let a = get("a");
        get("bb");
        assert!(Rc::ptr_eq(&a, &get("a")));
        assert!(Rc::ptr_eq(&a, &get("a")));
        assert_eq!(builds.get(), 2);
    }

    #[test]
    fn a_full_cache_still_serves_the_last_value() {
        let cache = ThreadCache::<String, usize>::default();
        let mut last = None;
        for i in 0..=CACHE_CAPACITY {
            last = Some(cache.get_or_build(&i.to_string(), || Ok(i)).unwrap());
        }
        assert!(cache.len() <= CACHE_CAPACITY);
        let served = cache
            .get_or_build(&CACHE_CAPACITY.to_string(), || unreachable!())
            .unwrap();
        assert!(Rc::ptr_eq(&last.unwrap(), &served));
    }

    #[test]
    fn a_shared_cache_builds_each_key_once() {
        let cache = SharedCache::<String, usize>::new();
        let builds = Cell::new(0);
        let get = |key: &str| {
            cache
                .with_or_build(
                    key,
                    || {
                        builds.set(builds.get() + 1);
                        Ok(key.len())
                    },
                    |&len| Ok(len),
                )
                .unwrap()
        };

        assert_eq!(get("a"), 1);
        assert_eq!(get("bb"), 2);
        assert_eq!(get("a"), 1);
        assert_eq!(builds.get(), 2);
    }

    #[test]
    fn a_full_shared_cache_starts_over_rather_than_growing() {
        let cache = SharedCache::<String, usize>::new();
        for i in 0..=CACHE_CAPACITY {
            cache
                .with_or_build(&i.to_string(), || Ok(i), |_| Ok(()))
                .unwrap();
        }
        assert!(cache.len() <= CACHE_CAPACITY);
    }
}
