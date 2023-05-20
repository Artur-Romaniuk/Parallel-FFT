#![feature(test)]

pub mod fft;

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{iterative::Iterative, parallel::Parallel, recursive::Recursive, Fft};
    use approx::assert_relative_eq;
    use num::Complex;
    use rand::{distributions::Standard, Rng};
    use rustfft::{algorithm::Radix4, Fft as OtherFft, FftDirection};
    use test::Bencher;
    const EPSILON: f64 = 10e-6;

    #[test]
    fn rec_small() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(256)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Recursive::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(256, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }

    #[test]
    fn rec_big() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Recursive::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(65536, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }

    #[test]
    fn iterative_small() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(256)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Iterative::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(256, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }

    #[test]
    fn iterative_big() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Iterative::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(65536, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }

    #[test]
    fn parallel_small() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(256)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Parallel::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(256, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }

    #[test]
    fn parallel_big() {
        let input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        let mut output = input.clone();
        Parallel::fft(&mut output);

        let mut refference_fft = input.clone();
        Radix4::new(65536, FftDirection::Forward).process(&mut refference_fft);

        output.iter().enumerate().for_each(|(index, val)| {
            assert_relative_eq!(
                val.norm_sqr(),
                refference_fft[index].norm_sqr(),
                epsilon = EPSILON
            );
        });
    }
    #[bench]
    fn bench_iterative_512(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(512)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_1024(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1024)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_2048(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(2048)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_4096(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(4096)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_16384(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(16384)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_65536(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_iterative_1048576(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1048576)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Iterative::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_512(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(512)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_1024(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1024)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_2048(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(2048)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_4096(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(4096)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_16384(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(16384)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_65536(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_recursive_1048576(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1048576)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Recursive::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_512(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(512)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_1024(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1024)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_2048(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(2048)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_4096(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(4096)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_16384(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(16384)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_65536(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(65536)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }

    #[bench]
    fn bench_parallel_1048576(b: &mut Bencher) {
        let mut input: Vec<Complex<f64>> = rand::thread_rng()
            .sample_iter(Standard)
            .take(1048576)
            .map(|(re, im)| Complex::new(re, im))
            .collect();

        b.iter(|| test::black_box(Parallel::fft(&mut input)));
    }
}
