use num::complex::Complex;
// use num_cpus;
// use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::f64::consts::PI;

pub struct Parallel {}
use super::Fft;

impl Parallel {
    fn bit_reverse(input: &mut [Complex<f64>]) {
        let n = input.len();
        if n <= 1 {
            return;
        }

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

impl Fft for Parallel {
    fn fft(input: &mut [Complex<f64>]) {
        if input.len() <= 1 {
            panic!("Input size must be greater than 1");
        }
        if input.len() & (input.len() - 1) != 0 {
            panic!("Input size must be a power of 2");
        }

        Self::bit_reverse(input);

        // Perform Cooley-Tukey decimation-in-time FFT using Rayon parallelization
        let mut size = 2;
        while size <= input.len() {
            let half_size = size / 2;
            let angle_step = 2.0 * PI / size as f64;

            input.par_chunks_mut(size).for_each(|chunk| {
                for i in (0..half_size).step_by(1) {
                    let angle = i as f64 * angle_step;
                    let twiddle_factor = Complex::from_polar(1.0, -angle);

                    let temp = twiddle_factor * chunk[i + half_size];
                    let chunk_i = chunk[i];
                    chunk[i] = chunk_i + temp;
                    chunk[i + half_size] = chunk_i - temp;
                }
            });

            size *= 2;
        }
    }
}
