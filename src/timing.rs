//! Optional per-stage build timings.
//!
//! Off by default. Enable at runtime with the `TACHIOM_TIMINGS` environment variable
//! (any value other than `0`, `false` or the empty string), or programmatically with
//! [`set_enabled`] — which is what the `--timings` flag on `tachiom_build` does:
//!
//! ```text
//! TACHIOM_TIMINGS=1 ./target/release/tachiom_build ...
//! ./target/release/tachiom_build --timings ...
//! ```
//!
//! Each enabled stage prints one `[timing] <stage>: <elapsed>` line, so a build log can be
//! reduced to its stage breakdown with `grep '\[timing\]'`.
//!
//! Output goes to stdout via `println!`. Through the Python bindings that means file
//! descriptor 1 rather than `sys.stdout`, so the lines are invisible to Jupyter cell capture,
//! `contextlib.redirect_stdout` and pytest's `capsys` — in a notebook they surface on the
//! terminal that started the kernel.
//!
//! Cost when disabled is one `Instant::now()` plus one relaxed atomic load per stage — a
//! handful of nanoseconds across a build that runs for hours.

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(UNSET);

/// Turn stage timings on or off, overriding `TACHIOM_TIMINGS`.
pub fn set_enabled(enabled: bool) {
    STATE.store(if enabled { ON } else { OFF }, Ordering::Relaxed);
}

/// Whether stage timings are currently reported.
///
/// On first call, falls back to the `TACHIOM_TIMINGS` environment variable.
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        UNSET => {
            let on = match std::env::var("TACHIOM_TIMINGS") {
                Ok(v) => !matches!(v.as_str(), "" | "0" | "false"),
                Err(_) => false,
            };
            STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
        state => state == ON,
    }
}

/// Print `[timing] <stage>: <elapsed since start>` if timings are enabled.
pub fn report(stage: &str, start: Instant) {
    if enabled() {
        println!("[timing] {}: {:.2?}", stage, start.elapsed());
    }
}
