use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

mod volatility;

/// Compute rolling Garman-Klass volatility from OHLC data.
///
/// This is a Zero-Copy FFI call: Python passes NumPy arrays directly
/// to Rust memory without copying. The caller MUST ensure arrays are
/// C-contiguous (use np.ascontiguousarray before calling).
///
/// Args:
///     open, high, low, close: Price arrays (must be same length).
///     window: Rolling window size.
///     ann_factor: Annualization factor (1095 for 8h periods).
///
/// Returns:
///     NumPy array of annualized volatility (NaN for warm-up periods).
#[pyfunction]
fn garman_klass_volatility<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    window: usize,
    ann_factor: f64,
) -> PyResult<pyo3::Bound<'py, PyArray1<f64>>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    if o.len() != h.len() || h.len() != l.len() || l.len() != c.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All arrays must have the same length",
        ));
    }

    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Window must be > 0",
        ));
    }

    let result = volatility::garman_klass_rolling(o, h, l, c, window, ann_factor);
    Ok(PyArray1::from_vec(py, result))
}

#[pymodule]
fn crypto_volatility_rust(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(garman_klass_volatility, m)?)?;
    Ok(())
}



