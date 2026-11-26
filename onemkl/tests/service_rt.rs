// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "link-rt")]

//! Single dynamic library (`link-rt`): choose the layers at run time.
//! Process-global and order-dependent, so this is its own test binary
//! with one test.

use onemkl::service::{self, InterfaceLayer, ThreadingLayer};

#[test]
fn rt_layers_can_be_selected_before_first_use() {
    // Must precede any other MKL call in this process.
    assert_eq!(
        service::set_interface_layer(InterfaceLayer::Lp64).unwrap(),
        InterfaceLayer::Lp64
    );
    assert_eq!(
        service::set_threading_layer(ThreadingLayer::Sequential).unwrap(),
        ThreadingLayer::Sequential
    );
    // The sequential layer reports a single thread.
    assert_eq!(service::max_threads(), 1);
    // And MKL still computes through the runtime-selected layers.
    let v = service::version_string();
    assert!(!v.is_empty());
}
