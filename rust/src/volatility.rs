/// Rolling Garman-Klass volatility estimator.
///
/// Assumes continuous Brownian motion (which is violated during
/// crypto liquidation cascades — documented as a known limitation).
/// The caller is responsible for passing C-contiguous arrays
/// (use np.ascontiguousarray in Python).

pub fn garman_klass_rolling(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    ann_factor: f64,
) -> Vec<f64> {
    let n = open.len();
    let mut result = vec![f64::NAN; n];
    let ln2 = (2.0_f64).ln();

    for i in (window - 1)..n {
        let mut sum = 0.0;
        for j in (i + 1 - window)..=i {
            let hl = (high[j] / low[j]).ln();
            let co = (close[j] / open[j]).ln();
            sum += 0.5 * hl * hl - (2.0 * ln2 - 1.0) * co * co;
        }
        // Clamp to zero before sqrt (GK can produce negative variance
        // on very narrow candles where co² dominates hl²)
        let variance = (sum / window as f64).max(0.0);
        result[i] = variance.sqrt() * ann_factor.sqrt();
    }
    result
}



