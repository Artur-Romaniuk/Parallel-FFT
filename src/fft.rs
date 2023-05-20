pub mod iterative;
pub mod parallel;
pub mod recursive;

use num::complex::Complex;

pub trait Fft {
    fn fft(vec: &mut [Complex<f64>]);
}

// fn bit_reverse_copy(a: &mut [Complex<f64>]) {
//     let n = a.len();

//     let bits = (n as f64).log2() as usize;

//     for k in 0..n {
//         let rev_k = k.reverse_bits() & ((1 << bits) - 1);
//         a[rev_k] = a[k];
//     }
// }

// pub fn iterative_fft(a: &mut [Complex<f64>]) {
//     let n = a.len();
//     bit_reverse_copy(a);

//     for s in 1..((n as f64).log2() as u32 + 1) {
//         let m = 1 << s;
//         let omega_m = Complex::new(0.0, -2.0 * std::f64::consts::PI / m as f64).exp();

//         for k in (0..n).step_by(m as usize) {
//             let mut omega = Complex::new(1.0, 0.0);

//             for j in 0..(m / 2) {
//                 let t = omega * a[k + j + m / 2];
//                 let u = a[k + j];
//                 a[k + j] = u + t;
//                 a[k + j + m / 2] = u - t;
//                 omega *= omega_m;
//             }
//         }
//     }
// }
