use num::complex::Complex;
use std::f64::consts::PI;

pub struct Iterative {}
use super::Fft;

impl Iterative {
    fn bit_reverse(input: &mut [Complex<f64>]) {
        let n = input.len();

        // Rearrange input vector
        let mut j = 0;
        for i in 1..n - 1 {
            let mut k = n / 2;
            while k <= j {
                j -= k;
                k /= 2;
            }
            j += k;

            if i < j {
                input.swap(i, j);
            }
        }
    }
}

impl Fft for Iterative {
    fn fft(input: &mut [Complex<f64>]) {
        if input.len() <= 1 {
            panic!("Input size must be greater than 1");
        }
        if input.len() & (input.len() - 1) != 0 {
            panic!("Input size must be a power of 2");
        }

        Self::bit_reverse(input);

        // Perform Cooley-Tukey decimation-in-time FFT
        let mut size = 2;
        while size <= input.len() {
            let half_size = size / 2;
            let angle_step = 2.0 * PI / size as f64;

            for chunk in input.chunks_mut(size) {
                for i in 0..half_size {
                    let angle = i as f64 * angle_step;
                    let twiddle_factor = Complex::from_polar(1.0, -angle);

                    let temp = twiddle_factor * chunk[i + half_size];
                    let chunk_i = chunk[i];
                    chunk[i] = chunk_i + temp;
                    chunk[i + half_size] = chunk_i - temp;
                }
            }

            size *= 2;
        }
    }
}
