use num::complex::Complex;
use std::f64::consts::PI;

pub struct Recursive {}
use super::Fft;


impl Recursive {
    fn fft_recursive(x: &mut [Complex<f64>]) {
         let n = x.len();
    if n <= 1 {
        return;
    }
    
    let mut even: Vec<Complex<f64>> = Vec::with_capacity(n / 2);
    let mut odd: Vec<Complex<f64>> = Vec::with_capacity(n / 2);
    
    for i in (0..n).step_by(2) {
        even.push(x[i]);
        odd.push(x[i + 1]);
    }
    
    Self::fft_recursive(&mut even);
    Self::fft_recursive(&mut odd);
    
    for k in 0..(n / 2) {
        let t = Complex::new(0.0, -2.0 * PI * k as f64 / n as f64).exp() * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
    }
}
}
impl Fft for Recursive {
    fn fft(input: &mut [Complex<f64>]) {
        if input.len() <= 1 {
            panic!("Input size must be greater than 1");
        }
        if input.len() & (input.len() - 1) != 0 {
            panic!("Input size must be a power of 2");
        }
        Recursive::fft_recursive(input);
    }
}
