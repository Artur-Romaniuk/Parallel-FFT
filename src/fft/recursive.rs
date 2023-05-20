use num::complex::Complex;
use std::f64::consts::PI;

pub struct Recursive {}
use super::Fft;
impl Recursive {
    fn fft_recursive(input: &mut [Complex<f64>]) {
        if input.len() <= 1 {
            return;
        }

        let n = input.len();

        let (even, odd): (Vec<_>, Vec<_>) = input.iter().enumerate().partition(|(i, _)| i % 2 == 0);

        let even_values: Vec<_> = even.into_iter().map(|(_, v)| *v).collect();
        let odd_values: Vec<_> = odd.into_iter().map(|(_, v)| *v).collect();

        Recursive::fft_recursive(&mut even_values.clone());
        Recursive::fft_recursive(&mut odd_values.clone());

        let angle_step = -2.0 * PI / n as f64;
        for k in 0..n / 2 {
            let twiddle_factor = Complex::from_polar(1.0, angle_step * k as f64);
            let t = twiddle_factor * odd_values[k];
            input[k] = even_values[k] + t;
            input[n / 2 + k] = even_values[k] - t;
        }
    }
}

impl Fft for Recursive {
    fn fft(input: &mut [Complex<f64>]) {
        if input.len() <= 1 {
            panic!("Input size must be greater than 1");
        }
        if input.len() & input.len() - 1 != 0 {
            panic!("Input size must be a power of 2");
        }
        Self::fft_recursive(input);
    }
}
