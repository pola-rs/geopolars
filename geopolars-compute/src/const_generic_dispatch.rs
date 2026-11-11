//! [with_const_flags](crate::with_const_flags),
//! and the enums it can dispatch on, declared by [const_enum](crate::const_enum).

/// Dispatches runtime `bool` flags and [const_enum](crate::const_enum) values
/// into generic implementations,
/// so that the hot path never branches on them.
/// `with_const_flags!(A = a, M: Mode = m; f::<A, M>(args))` calls `f::<A, M>(args)`,
/// with `a` turned into a const `bool` named `A`,
/// and `m` into a const `u8` named `M` that `Mode::from_u8` turns back into a `Mode`.
#[macro_export]
macro_rules! with_const_flags {
    // `$rest` is a single token tree,
    // as a repetition cannot be used inside the repetition over the variants.
    (@mode $name:ident = $mode:expr; [$($variant:path),+]; $rest:tt) => {
        match $mode {
            $($variant => {
                const $name: u8 = $variant as u8;
                $crate::with_const_flags! $rest
            })+
        }
    };
    (@mode $name:ident = $mode:expr; [$($variant:path),+]; ; $call:expr) => {
        match $mode {
            $($variant => {
                const $name: u8 = $variant as u8;
                $call
            })+
        }
    };
    ($name:ident : $enum:ident = $mode:expr , $($rest:tt)+) => {
        $enum!($name = $mode; ($($rest)+))
    };
    ($name:ident : $enum:ident = $mode:expr ; $call:expr) => {
        $enum!($name = $mode; ; $call)
    };
    ($name:ident = $flag:expr , $($rest:tt)+) => {
        if $flag {
            const $name: bool = true;
            $crate::with_const_flags!($($rest)+)
        } else {
            const $name: bool = false;
            $crate::with_const_flags!($($rest)+)
        }
    };
    ($name:ident = $flag:expr ; $call:expr) => {
        if $flag {
            const $name: bool = true;
            $call
        } else {
            const $name: bool = false;
            $call
        }
    };
}

/// Declares a fieldless enum that [with_const_flags](crate::with_const_flags) can dispatch on,
/// as `M: Mode = m`.
/// A kernel takes it as `const M: u8`, and matches on `const { Mode::from_u8(M) }`.
///
/// The dispatch only works within the crate that declares the enum.
#[macro_export]
macro_rules! const_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $enum:ident { $($(#[$variant_meta:meta])* $variant:ident),+ $(,)? }
    ) => {
        $crate::const_enum!(
            @with_dollar ($)
            $(#[$meta])*
            $vis enum $enum { $($(#[$variant_meta])* $variant),+ }
        );
    };
    // The generated macro needs a literal `$`, and writing `$$` is unstable.
    (
        @with_dollar ($d:tt)
        $(#[$meta:meta])*
        $vis:vis enum $enum:ident { $($(#[$variant_meta:meta])* $variant:ident),+ }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(u8)]
        $vis enum $enum { $($(#[$variant_meta])* $variant),+ }

        impl $enum {
            /// Panics if `v` is not the `u8` of a variant,
            /// which is a compile error when evaluated in a `const` block.
            pub const fn from_u8(v: u8) -> Self {
                $(if v == $enum::$variant as u8 {
                    return $enum::$variant;
                })+
                panic!(concat!("invalid ", stringify!($enum)))
            }
        }

        #[allow(unused_macros)]
        macro_rules! const_enum_variants {
            ($d name:ident = $d mode:expr; $d ($d rest:tt)*) => {
                $crate::with_const_flags!(
                    @mode $d name = $d mode; [$($enum::$variant),+]; $d ($d rest)*
                )
            };
        }
        // Being only a macro, this cannot clash with the enum of the same name.
        // A later `const_enum!` in the same module shadows `const_enum_variants`,
        // but this import keeps the definition above.
        #[allow(unused_imports)]
        pub(crate) use const_enum_variants as $enum;
    };
}

#[cfg(test)]
mod tests {
    //! The Python suite cannot reach [const_enum](crate::const_enum)
    //! until a kernel dispatches on one.

    mod modes {
        const_enum!(
            pub enum Mode {
                A,
                B,
                Neither,
            }
        );
        const_enum!(
            pub enum Side {
                Left,
                Right,
            }
        );
    }

    use modes::{Mode, Side};

    fn kernel<const Z: bool, const M: u8, const S: u8>() -> (bool, Mode, Side) {
        (Z, const { Mode::from_u8(M) }, const { Side::from_u8(S) })
    }

    #[test]
    fn with_const_flags_dispatches_bools_and_enums_together() {
        for z in [false, true] {
            for mode in [Mode::A, Mode::B, Mode::Neither] {
                for side in [Side::Left, Side::Right] {
                    let dispatched = with_const_flags!(
                        M: Mode = mode, Z = z, S: Side = side;
                        kernel::<Z, M, S>()
                    );
                    assert_eq!(dispatched, (z, mode, side));
                }
            }
        }
    }
}
