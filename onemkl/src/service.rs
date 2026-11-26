//! Service routines: version info, threading control, memory tracking,
//! verbose mode, and overall library lifecycle.

use core::ffi::{c_char, c_int};
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::slice;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::ffi::{CStr, CString};
use std::sync::Mutex;

use onemkl_sys as sys;

use crate::error::{Error, Result};

// =====================================================================
// Version
// =====================================================================

/// Information about the running oneMKL library version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major version (e.g. `2025` for MKL 2025.x).
    pub major: i32,
    /// Minor version.
    pub minor: i32,
    /// Update / patch number.
    pub update: i32,
    /// Build identifier (e.g. `"20251007"`).
    pub build: String,
    /// Targeted processor architecture (e.g. `"Intel(R) Architecture processors"`).
    pub processor: String,
    /// Platform string (e.g. `"Intel(R) 64 architecture"`).
    pub platform: String,
}

/// Query the oneMKL version.
pub fn version() -> Version {
    let mut v = sys::MKLVersion::default();
    unsafe { sys::MKL_Get_Version(&mut v) };
    Version {
        major: v.MajorVersion,
        minor: v.MinorVersion,
        update: v.UpdateVersion,
        build: c_str_to_owned(v.Build),
        processor: c_str_to_owned(v.Processor),
        platform: c_str_to_owned(v.Platform),
    }
}

/// Single-line description of the running oneMKL library, as returned
/// by `MKL_Get_Version_String`.
pub fn version_string() -> String {
    let mut buf = vec![0u8; 256];
    unsafe {
        sys::MKL_Get_Version_String(buf.as_mut_ptr().cast::<c_char>(), buf.len() as c_int);
    }
    if let Some(end) = buf.iter().position(|&b| b == 0) {
        buf.truncate(end);
    }
    let trimmed: Vec<u8> = buf.into_iter().rev().skip_while(|&b| b == b' ').collect();
    let mut out: Vec<u8> = trimmed.into_iter().rev().collect();
    out.retain(|&b| b != 0);
    String::from_utf8_lossy(&out).into_owned()
}

// =====================================================================
// Threading
// =====================================================================

/// Set the global thread count for oneMKL routines.
pub fn set_num_threads(n: i32) {
    unsafe { sys::MKL_Set_Num_Threads(n) };
}

/// Set the thread count for the current thread only. Returns the
/// previous local thread count for that thread.
pub fn set_num_threads_local(n: i32) -> i32 {
    unsafe { sys::MKL_Set_Num_Threads_Local(n) }
}

/// Scoped guard that restores the previous **local** thread count when
/// dropped. Useful for "during this MKL call use N threads, then put
/// it back" patterns in inference loops.
///
/// ```no_run
/// use onemkl::service::ThreadCountGuard;
///
/// let _g = ThreadCountGuard::new(1);
/// // ... MKL calls in this scope run single-threaded ...
/// ```
pub struct ThreadCountGuard {
    previous: i32,
}

impl ThreadCountGuard {
    /// Set the calling thread's local MKL thread count to `n` and
    /// remember the previous value.
    pub fn new(n: i32) -> Self {
        let previous = unsafe { sys::MKL_Set_Num_Threads_Local(n) };
        Self { previous }
    }
}

impl Drop for ThreadCountGuard {
    fn drop(&mut self) {
        unsafe { sys::MKL_Set_Num_Threads_Local(self.previous) };
    }
}

/// Maximum number of threads oneMKL is allowed to use.
pub fn max_threads() -> i32 {
    unsafe { sys::MKL_Get_Max_Threads() }
}

/// Enable or disable dynamic adjustment of the number of threads
/// (`MKL_Set_Dynamic`).
pub fn set_dynamic(flag: bool) {
    unsafe { sys::MKL_Set_Dynamic(if flag { 1 } else { 0 }) };
}

/// Read the current dynamic-threading setting.
pub fn dynamic() -> bool {
    unsafe { sys::MKL_Get_Dynamic() != 0 }
}

// =====================================================================
// Verbose mode
// =====================================================================

/// Enable or disable verbose mode (`MKL_Verbose`). When enabled, oneMKL
/// prints a one-line trace of every supported routine call.
pub fn set_verbose(flag: bool) {
    unsafe { sys::MKL_Verbose(if flag { 1 } else { 0 }) };
}

/// Redirect verbose output to the given file path. Pass `None` to
/// reset to stdout.
pub fn set_verbose_output_file(path: Option<&str>) -> Result<()> {
    let status = match path {
        Some(p) => {
            let cstr = CString::new(p)
                .map_err(|_| Error::InvalidArgument("verbose output path contains a NUL byte"))?;
            unsafe { sys::MKL_Verbose_Output_File(cstr.as_ptr()) }
        }
        None => unsafe { sys::MKL_Verbose_Output_File(core::ptr::null()) },
    };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "MKL_Verbose_Output_File failed (path inaccessible?)",
        ))
    }
}

// =====================================================================
// Memory
// =====================================================================

/// Snapshot of MKL's internal allocator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemStat {
    /// Total bytes allocated by MKL.
    pub bytes_allocated: i64,
    /// Number of distinct buffers currently held.
    pub num_buffers: i32,
}

/// Read the current allocation statistics.
pub fn mem_stat() -> MemStat {
    let mut nbuffers: c_int = 0;
    let bytes = unsafe { sys::MKL_Mem_Stat(&mut nbuffers) };
    MemStat {
        bytes_allocated: bytes,
        num_buffers: nbuffers,
    }
}

/// Peak memory usage in bytes since the last reset. Pass `reset=true`
/// to clear the high-water mark for subsequent calls.
pub fn peak_mem_usage(reset: bool) -> i64 {
    let r: c_int = if reset { 1 } else { 0 };
    unsafe { sys::MKL_Peak_Mem_Usage(r) }
}

/// Free internal MKL buffers (best-effort cleanup).
pub fn free_buffers() {
    unsafe { sys::MKL_Free_Buffers() };
}

/// Free per-thread MKL buffers for the current thread.
pub fn thread_free_buffers() {
    unsafe { sys::MKL_Thread_Free_Buffers() };
}

/// Set a soft memory limit for MKL allocations. `mem_type` is reserved
/// (currently 0 for total).
pub fn set_memory_limit(mem_type: i32, limit_bytes: usize) -> Result<()> {
    let status = unsafe { sys::MKL_Set_Memory_Limit(mem_type, limit_bytes) };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "MKL_Set_Memory_Limit rejected the requested limit",
        ))
    }
}

// =====================================================================
// Aligned allocations
// =====================================================================

/// Owned buffer of `T`s, allocated by `MKL_malloc` with the requested
/// alignment and freed by `MKL_free` on drop.
///
/// SIMD-aligned backing storage is the usual reason to reach for this:
/// 64-byte alignment matches AVX-512 cache-line loads; 32 byte matches
/// AVX2. For raw tensor data feeding an MKL routine, allocating here
/// avoids a copy compared to `Vec<T>` (which only guarantees `align_of::<T>()`).
///
/// The buffer is zero-initialized via `MKL_malloc`'s contract; treat
/// the contents as uninitialized only if you skip the value-writing
/// constructors and explicitly write `MaybeUninit<T>` instead.
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: `MKL_malloc` returns a heap pointer that's not tied to any
// thread; the buffer is `Send`/`Sync` to the same extent `Box<[T]>` is.
unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> AlignedBuffer<T> {
    /// Allocate `len` elements of `T` with the given alignment in
    /// bytes. `alignment` must be a power of two and a multiple of
    /// `align_of::<T>()`.
    ///
    /// Returns `Err(Error::AllocationFailure)` if `MKL_malloc` returns
    /// null (out of memory).
    pub fn new(len: usize, alignment: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        let mut buf = Self::new_uninit(len, alignment)?;
        let default = T::default();
        for slot in buf.as_mut_slice().iter_mut() {
            *slot = default;
        }
        Ok(buf)
    }

    /// Allocate without initializing the contents. Callers must write
    /// every element before reading.
    pub fn new_uninit(len: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(Error::InvalidArgument("alignment must be a power of two"));
        }
        if alignment < core::mem::align_of::<T>() {
            return Err(Error::InvalidArgument(
                "alignment must be a multiple of align_of::<T>()",
            ));
        }
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(Error::InvalidArgument("len * size_of::<T>() overflowed"))?;
        let ptr = unsafe { sys::MKL_malloc(bytes, alignment as c_int) };
        if ptr.is_null() {
            return Err(Error::AllocationFailure);
        }
        // SAFETY: pointer is non-null per the check above.
        let ptr = unsafe { NonNull::new_unchecked(ptr.cast::<T>()) };
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer holds zero elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow as an immutable slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: pointer is valid for self.len elements for the
        // lifetime of `self`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrow as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: pointer is valid for self.len elements for the
        // lifetime of `self`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Raw pointer to the first element.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer to the first element.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: pointer came from MKL_malloc and we own the allocation.
        unsafe { sys::MKL_free(self.ptr.as_ptr().cast()) };
    }
}

impl<T> core::ops::Deref for AlignedBuffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> core::ops::DerefMut for AlignedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// =====================================================================
// CPU / ISA dispatch
// =====================================================================

/// Vector ISA level. Pass to [`enable_instructions`] to force MKL to
/// dispatch to a specific code path (e.g. for benchmarking, or to work
/// around a buggy fallback).
///
/// MKL has gradually removed dispatch entries for older or
/// Xeon-Phi-only ISAs (original AVX, AVX-512-MIC) — newer headers
/// (≥ MKL 2026) drop those symbols entirely, so they're not exposed
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaLevel {
    /// SSE4.2 (baseline x86-64-v2).
    Sse42,
    /// AVX2.
    Avx2,
    /// AVX2 enabled-1 (atom-style extension).
    Avx2E1,
    /// AVX-512 (skylake-class).
    Avx512,
    /// AVX-512 enabled-1 (cascade lake VNNI).
    Avx512E1,
    /// AVX-512 enabled-2 (cooper lake bf16).
    Avx512E2,
    /// AVX-512 enabled-3 (sapphire rapids).
    Avx512E3,
    /// AVX-512 enabled-4.
    Avx512E4,
    /// AVX-512 enabled-5.
    Avx512E5,
    /// AVX10.
    Avx10,
}

impl IsaLevel {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Sse42 => sys::MKL_ENABLE_SSE4_2,
            Self::Avx2 => sys::MKL_ENABLE_AVX2,
            Self::Avx2E1 => sys::MKL_ENABLE_AVX2_E1,
            Self::Avx512 => sys::MKL_ENABLE_AVX512,
            Self::Avx512E1 => sys::MKL_ENABLE_AVX512_E1,
            Self::Avx512E2 => sys::MKL_ENABLE_AVX512_E2,
            Self::Avx512E3 => sys::MKL_ENABLE_AVX512_E3,
            Self::Avx512E4 => sys::MKL_ENABLE_AVX512_E4,
            Self::Avx512E5 => sys::MKL_ENABLE_AVX512_E5,
            Self::Avx10 => sys::MKL_ENABLE_AVX10,
        };
        v as c_int
    }
}

/// Pin oneMKL's dispatch to a specific vector-ISA tier. Returns `Ok(())`
/// if the requested level is supported and now active.
///
/// Call **before** any MKL routine to take effect — MKL caches the
/// dispatched code path on first use.
pub fn enable_instructions(level: IsaLevel) -> Result<()> {
    let r = unsafe { sys::MKL_Enable_Instructions(level.as_int()) };
    // MKL returns 1 on success, 0 on failure (e.g. CPU doesn't
    // support the requested ISA).
    if r == 1 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "MKL_Enable_Instructions: CPU does not support the requested ISA level",
        ))
    }
}

/// Interface layer of the single dynamic library (`link-rt`): the integer
/// width (and, for [`GnuLp64`](InterfaceLayer::GnuLp64), the GNU Fortran
/// calling convention) `mkl_rt` uses.
#[cfg(feature = "link-rt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceLayer {
    /// 32-bit `MKL_INT`, Intel Fortran conventions.
    Lp64,
    /// 64-bit `MKL_INT`.
    Ilp64,
    /// 32-bit `MKL_INT`, GNU Fortran conventions.
    GnuLp64,
}

/// Threading layer of the single dynamic library (`link-rt`).
#[cfg(feature = "link-rt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadingLayer {
    /// Intel OpenMP runtime.
    IntelOpenMp,
    /// No internal threading.
    Sequential,
    /// GNU OpenMP runtime.
    GnuOpenMp,
    /// oneTBB.
    Tbb,
}

/// Select `mkl_rt`'s interface layer (`MKL_Set_Interface_Layer`). Call it
/// **before** any other MKL routine; afterwards MKL ignores it. The layer
/// must match the `lp64`/`ilp64` feature the bindings were built with.
/// Returns the layer MKL reports as set.
#[cfg(feature = "link-rt")]
pub fn set_interface_layer(layer: InterfaceLayer) -> Result<InterfaceLayer> {
    let code = match layer {
        InterfaceLayer::Lp64 => sys::MKL_INTERFACE_LP64,
        InterfaceLayer::Ilp64 => sys::MKL_INTERFACE_ILP64,
        InterfaceLayer::GnuLp64 => sys::MKL_INTERFACE_LP64 | sys::MKL_INTERFACE_GNU,
    } as c_int;
    let r = unsafe { sys::MKL_Set_Interface_Layer(code) };
    match r {
        x if x == sys::MKL_INTERFACE_LP64 as c_int => Ok(InterfaceLayer::Lp64),
        x if x == sys::MKL_INTERFACE_ILP64 as c_int => Ok(InterfaceLayer::Ilp64),
        x if x == (sys::MKL_INTERFACE_LP64 | sys::MKL_INTERFACE_GNU) as c_int => {
            Ok(InterfaceLayer::GnuLp64)
        }
        _ => Err(Error::InvalidArgument(
            "MKL_Set_Interface_Layer: the layer could not be set (already initialised?)",
        )),
    }
}

/// Select `mkl_rt`'s threading layer (`MKL_Set_Threading_Layer`). Call it
/// **before** any other MKL routine. Returns the layer MKL reports as set.
#[cfg(feature = "link-rt")]
pub fn set_threading_layer(layer: ThreadingLayer) -> Result<ThreadingLayer> {
    let code = match layer {
        ThreadingLayer::IntelOpenMp => sys::MKL_THREADING_INTEL,
        ThreadingLayer::Sequential => sys::MKL_THREADING_SEQUENTIAL,
        ThreadingLayer::GnuOpenMp => sys::MKL_THREADING_GNU,
        ThreadingLayer::Tbb => sys::MKL_THREADING_TBB,
    } as c_int;
    let r = unsafe { sys::MKL_Set_Threading_Layer(code) };
    match r {
        x if x == sys::MKL_THREADING_INTEL as c_int => Ok(ThreadingLayer::IntelOpenMp),
        x if x == sys::MKL_THREADING_SEQUENTIAL as c_int => Ok(ThreadingLayer::Sequential),
        x if x == sys::MKL_THREADING_GNU as c_int => Ok(ThreadingLayer::GnuOpenMp),
        x if x == sys::MKL_THREADING_TBB as c_int => Ok(ThreadingLayer::Tbb),
        _ => Err(Error::InvalidArgument(
            "MKL_Set_Threading_Layer: the layer could not be set (unavailable or already initialised?)",
        )),
    }
}

/// Read the CPU's current clock counter (TSC-like). Useful for fine-
/// grained timing of MKL routines.
pub fn cpu_clocks() -> u64 {
    let mut clocks: u64 = 0;
    unsafe { sys::MKL_Get_Cpu_Clocks(&mut clocks) };
    clocks
}

/// Current CPU frequency in GHz.
pub fn cpu_frequency_ghz() -> f64 {
    unsafe { sys::MKL_Get_Cpu_Frequency() }
}

/// Maximum CPU frequency in GHz.
pub fn max_cpu_frequency_ghz() -> f64 {
    unsafe { sys::MKL_Get_Max_Cpu_Frequency() }
}

// =====================================================================
// Conditional numerical reproducibility (CBWR)
// =====================================================================

/// A conditional-numerical-reproducibility (CBWR) code branch.
///
/// Branches older MKL releases offered but oneMKL 2026 headers dropped
/// (SSE3, SSSE3, SSE4.1, AVX, Xeon Phi) are not exposed.
///
/// Pinning a branch makes oneMKL produce bit-identical results across
/// CPUs that support it (and across thread counts, when set with
/// [`cbwr_set_strict`]), at some cost in speed. See [`cbwr_set`] and [`cbwr_set_strict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CbwrBranch {
    /// CBWR off (the default): MKL picks the fastest branch.
    Off,
    /// The best branch for this CPU, chosen automatically.
    Auto,
    /// Compatible branch.
    Compatible,
    /// SSE2.
    Sse2,
    /// SSE4.2.
    Sse42,
    /// AVX2.
    Avx2,
    /// AVX-512.
    Avx512,
    /// AVX-512 enabled-1.
    Avx512E1,
    /// AVX10.
    Avx10,
}

impl CbwrBranch {
    fn as_int(self) -> c_int {
        (match self {
            Self::Off => sys::MKL_CBWR_OFF,
            Self::Auto => sys::MKL_CBWR_AUTO,
            Self::Compatible => sys::MKL_CBWR_COMPATIBLE,
            Self::Sse2 => sys::MKL_CBWR_SSE2,
            Self::Sse42 => sys::MKL_CBWR_SSE4_2,
            Self::Avx2 => sys::MKL_CBWR_AVX2,
            Self::Avx512 => sys::MKL_CBWR_AVX512,
            Self::Avx512E1 => sys::MKL_CBWR_AVX512_E1,
            Self::Avx10 => sys::MKL_CBWR_AVX10,
        }) as c_int
    }

    fn from_int(v: c_int) -> Option<Self> {
        Some(match v as u32 {
            // `MKL_CBWR_Get(MKL_CBWR_BRANCH)` reports "CBWR off" (the default
            // state) as `MKL_CBWR_BRANCH_OFF`, not `MKL_CBWR_OFF`.
            sys::MKL_CBWR_OFF | sys::MKL_CBWR_BRANCH_OFF => Self::Off,
            sys::MKL_CBWR_AUTO => Self::Auto,
            sys::MKL_CBWR_COMPATIBLE => Self::Compatible,
            sys::MKL_CBWR_SSE2 => Self::Sse2,
            sys::MKL_CBWR_SSE4_2 => Self::Sse42,
            sys::MKL_CBWR_AVX2 => Self::Avx2,
            sys::MKL_CBWR_AVX512 => Self::Avx512,
            sys::MKL_CBWR_AVX512_E1 => Self::Avx512E1,
            sys::MKL_CBWR_AVX10 => Self::Avx10,
            _ => return None,
        })
    }
}

/// Why [`cbwr_set`] / [`cbwr_set_strict`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CbwrError {
    /// MKL rejected the settings as invalid input
    /// (`MKL_CBWR_ERR_INVALID_INPUT`).
    InvalidInput,
    /// The branch is not supported by this CPU
    /// (`MKL_CBWR_ERR_UNSUPPORTED_BRANCH`).
    UnsupportedBranch,
    /// MKL has already dispatched, so the mode can no longer change
    /// (`MKL_CBWR_ERR_MODE_CHANGE_FAILURE`). A lifecycle error, not bad
    /// input: call before any other MKL routine.
    ModeLocked,
    /// Any other negative status from `MKL_CBWR_Set`.
    Other(i32),
}

impl core::fmt::Display for CbwrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput => f.write_str("MKL_CBWR_Set: invalid input"),
            Self::UnsupportedBranch => {
                f.write_str("MKL_CBWR_Set: branch not supported by this CPU")
            }
            Self::ModeLocked => {
                f.write_str("MKL_CBWR_Set: mode can no longer change (MKL already dispatched)")
            }
            Self::Other(c) => write!(f, "MKL_CBWR_Set: status {c}"),
        }
    }
}

impl std::error::Error for CbwrError {}

fn cbwr_set_mode(mut mode: c_int, strict: bool) -> core::result::Result<(), CbwrError> {
    if strict {
        mode |= sys::MKL_CBWR_STRICT as c_int;
    }
    match unsafe { sys::MKL_CBWR_Set(mode) } {
        0 => Ok(()),
        sys::MKL_CBWR_ERR_INVALID_INPUT => Err(CbwrError::InvalidInput),
        sys::MKL_CBWR_ERR_UNSUPPORTED_BRANCH => Err(CbwrError::UnsupportedBranch),
        sys::MKL_CBWR_ERR_MODE_CHANGE_FAILURE => Err(CbwrError::ModeLocked),
        r => Err(CbwrError::Other(r)),
    }
}

/// Select the CBWR branch (`MKL_CBWR_Set`). Results are reproducible
/// across CPUs that support the branch, for a fixed thread count; use
/// [`cbwr_set_strict`] to also fix them across thread counts.
///
/// Call **before** any other MKL routine: once MKL has dispatched, the
/// mode can no longer change and this returns [`CbwrError::ModeLocked`].
pub fn cbwr_set(branch: CbwrBranch) -> core::result::Result<(), CbwrError> {
    cbwr_set_mode(branch.as_int(), false)
}

/// Like [`cbwr_set`], but also strict: results are reproducible across
/// different thread counts (`MKL_CBWR_STRICT`).
pub fn cbwr_set_strict(branch: CbwrBranch) -> core::result::Result<(), CbwrError> {
    cbwr_set_mode(branch.as_int(), true)
}

/// The CBWR branch currently in effect (`MKL_CBWR_Get(MKL_CBWR_BRANCH)`).
pub fn cbwr_branch() -> Result<CbwrBranch> {
    let v = unsafe { sys::MKL_CBWR_Get(sys::MKL_CBWR_BRANCH as c_int) };
    CbwrBranch::from_int(v).ok_or(Error::InvalidArgument(
        "MKL_CBWR_Get: unrecognised branch code",
    ))
}

/// Whether strict CBWR (thread-count independence) is in effect.
pub fn cbwr_strict() -> bool {
    let v = unsafe { sys::MKL_CBWR_Get(sys::MKL_CBWR_ALL) };
    v > 0 && (v as u32 & sys::MKL_CBWR_STRICT) != 0
}

/// The branch `Auto` would pick on this CPU (`MKL_CBWR_Get_Auto_Branch`).
/// Measured on oneMKL 2026.2.1 (Linux x86_64) this returns
/// [`CbwrBranch::Auto`] itself, not a concrete branch.
pub fn cbwr_auto_branch() -> Result<CbwrBranch> {
    let v = unsafe { sys::MKL_CBWR_Get_Auto_Branch() };
    CbwrBranch::from_int(v).ok_or(Error::InvalidArgument(
        "MKL_CBWR_Get_Auto_Branch: unrecognised branch code",
    ))
}

// =====================================================================
// Error and progress handlers
// =====================================================================

/// A handler for MKL's `xerbla` error routine: called with the routine
/// name (for example `"DPOTRF"`) and the 1-based position of the illegal
/// argument (for example `2` for an illegal second argument). Measured
/// on oneMKL 2026.2.1 (Linux x86_64): `xerbla` gets the positive
/// position, while the routine's own `info` output is its negation (`-2`).
///
/// A handler that panics is caught and the panic is dropped: unwinding
/// into MKL would be undefined behaviour.
pub type XerblaHandler = fn(routine: &str, argument: i32);

static XERBLA_HANDLER: AtomicUsize = AtomicUsize::new(0);
/// What `mkl_set_xerbla` returned the first time we replaced it, so
/// `set_xerbla_handler(None)` can put MKL's own entry back. `None` =
/// we have not replaced it.
static XERBLA_ORIGINAL: Mutex<Option<sys::XerblaEntry>> = Mutex::new(None);

unsafe extern "C" fn xerbla_trampoline(name: *const c_char, num: *const c_int, len: c_int) {
    let h = XERBLA_HANDLER.load(Ordering::Acquire);
    if h == 0 {
        return;
    }
    // SAFETY: `h` was stored from an `XerblaHandler` by `set_xerbla_handler`.
    let handler: XerblaHandler = unsafe { core::mem::transmute::<usize, XerblaHandler>(h) };
    let routine = if name.is_null() || len <= 0 {
        String::new()
    } else {
        // SAFETY: MKL passes a Fortran-style (not NUL-terminated) name
        // of exactly `len` bytes.
        let bytes = unsafe { slice::from_raw_parts(name as *const u8, len as usize) };
        String::from_utf8_lossy(bytes)
            .trim_end_matches(|c: char| c == ' ' || c == '\0')
            .to_owned()
    };
    // SAFETY: MKL passes a valid pointer to the argument position.
    let argument = if num.is_null() { 0 } else { unsafe { *num } };
    let _ = std::panic::catch_unwind(|| handler(&routine, argument));
}

/// Install (`Some`) or remove (`None`) a process-global handler for MKL's
/// `xerbla` error routine (`mkl_set_xerbla`). With `None`, the entry MKL
/// had before our first install is put back.
///
/// The handler replaces MKL's default `xerbla`, which prints a message.
/// It can run on any thread MKL calls from. Only Fortran-style entry
/// points (for example `onemkl_sys::dpotrf`) report through `xerbla`; the
/// safe wrappers in this crate validate arguments first and do not.
pub fn set_xerbla_handler(handler: Option<XerblaHandler>) {
    let mut original = XERBLA_ORIGINAL.lock().unwrap_or_else(|e| e.into_inner());
    match handler {
        Some(h) => {
            XERBLA_HANDLER.store(h as usize, Ordering::Release);
            let previous = unsafe { sys::mkl_set_xerbla(Some(xerbla_trampoline)) };
            if original.is_none() {
                *original = Some(previous);
            }
        }
        None => {
            if let Some(entry) = original.take() {
                unsafe { sys::mkl_set_xerbla(entry) };
            }
            XERBLA_HANDLER.store(0, Ordering::Release);
        }
    }
}

/// What a [`ProgressHandler`] tells MKL to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressControl {
    /// Keep computing (the handler returns `0` to MKL).
    Continue,
    /// Ask MKL to stop (the handler returns non-zero to MKL). Measured
    /// on oneMKL 2026.2.1 (Linux x86_64): `dpotrf` stops after that
    /// callback and returns `info = -1002`.
    Stop,
}

/// A handler for MKL's progress callback: `thread` is the reporting
/// thread, `step` the progress counter, `stage` a description. A
/// handler that panics is caught and treated as [`ProgressControl::Continue`].
pub type ProgressHandler = fn(thread: i32, step: i32, stage: &str) -> ProgressControl;

static PROGRESS_HANDLER: AtomicUsize = AtomicUsize::new(0);
static PROGRESS_ORIGINAL: Mutex<Option<sys::ProgressEntry>> = Mutex::new(None);

unsafe extern "C" fn progress_trampoline(
    thread: *mut c_int,
    step: *mut c_int,
    stage: *mut c_char,
    stage_len: c_int,
) -> c_int {
    let h = PROGRESS_HANDLER.load(Ordering::Acquire);
    if h == 0 {
        return 0;
    }
    // SAFETY: `h` was stored from a `ProgressHandler` by `set_progress_handler`.
    let handler: ProgressHandler = unsafe { core::mem::transmute::<usize, ProgressHandler>(h) };
    let stage_str = if stage.is_null() || stage_len <= 0 {
        String::new()
    } else {
        // SAFETY: Fortran-style, `stage_len` bytes, not NUL-terminated.
        let bytes = unsafe { slice::from_raw_parts(stage as *const u8, stage_len as usize) };
        String::from_utf8_lossy(bytes)
            .trim_end_matches(|c: char| c == ' ' || c == '\0')
            .to_owned()
    };
    let t = if thread.is_null() {
        0
    } else {
        unsafe { *thread }
    };
    let st = if step.is_null() { 0 } else { unsafe { *step } };
    match std::panic::catch_unwind(|| handler(t, st, &stage_str)) {
        Ok(ProgressControl::Stop) => 1,
        _ => 0,
    }
}

/// Install (`Some`) or remove (`None`) a process-global progress handler
/// (`mkl_set_progress`). With `None`, the entry MKL had before our first
/// install is put back. It can run on any thread MKL computes on.
pub fn set_progress_handler(handler: Option<ProgressHandler>) {
    let mut original = PROGRESS_ORIGINAL.lock().unwrap_or_else(|e| e.into_inner());
    match handler {
        Some(h) => {
            PROGRESS_HANDLER.store(h as usize, Ordering::Release);
            let previous = unsafe { sys::mkl_set_progress(Some(progress_trampoline)) };
            if original.is_none() {
                *original = Some(previous);
            }
        }
        None => {
            if let Some(entry) = original.take() {
                unsafe { sys::mkl_set_progress(entry) };
            }
            PROGRESS_HANDLER.store(0, Ordering::Release);
        }
    }
}

// =====================================================================
// Library lifecycle
// =====================================================================

/// Tear down all internal oneMKL state. Call once at program exit if
/// running in an environment that needs deterministic cleanup.
pub fn finalize() {
    unsafe { sys::MKL_Finalize() };
}

// =====================================================================
// Helpers
// =====================================================================

#[inline]
fn c_str_to_owned(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: MKL guarantees these strings are NUL-terminated and live
    // for the program's lifetime.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}
