#![cfg(feature = "vm")]

//! onemkl never calls `vmlSetMode`, so every `onemkl::vm` function runs in
//! whatever accuracy mode MKL starts a thread with. Pin that mode: if MKL's
//! default ever changes, this fails and the crate docs must be revisited.

use onemkl_sys::{VML_ACCURACY_MASK, VML_HA, vmlGetMode};

#[test]
fn default_accuracy_mode_is_high_accuracy() {
    // SAFETY: vmlGetMode reads thread-local VML state and has no preconditions.
    let mode = unsafe { vmlGetMode() };
    assert_eq!(
        mode & VML_ACCURACY_MASK,
        VML_HA,
        "VML mode word = {mode:#x}"
    );
}
