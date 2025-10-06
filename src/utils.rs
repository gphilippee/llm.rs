use rand::{random, Rng};

pub fn check_tensor(pred: &[f32], target: &[f32], n: usize, label: &str) -> bool {
    let print_upto = 0;
    let mut ok = true;
    let mut maxdiff = 0f32;
    let tol = 2e-2f32;
    println!("{}", label);
    for i in 0..n {
        // look at the diffence at position i of these two tensors
        let diff = (pred[i] - target[i]).abs();

        // keep track of the overall error
        ok = ok & (diff <= tol);
        if diff > maxdiff {
            maxdiff = diff;
        }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if i < print_upto {
            if diff <= tol {
                if i < print_upto {
                    println!("OK ");
                }
            } else {
                if i < print_upto {
                    println!("NOT OK ");
                }
            }
            println!("{} {}", pred[i], target[i]);
        }
    }
    // print the final result for this tensor
    if ok {
        println!("TENSOR OK, maxdiff = {}", maxdiff);
    } else {
        println!("TENSOR NOT OK, maxdiff = {}", maxdiff);
    }
    return ok;
}

pub fn sample_mult(probs: &[f32], n: usize) -> usize {
    let mut cdf: f32 = 0.0;
    let random_prod = random::<f32>();
    for i in 0..n {
        cdf += probs[i];
        if cdf > random_prod {
            return i;
        }
    }
    n - 1
}

pub fn init_random_permutation(n: usize) -> Vec<usize> {
    let permutation = (0..n).collect::<Vec<usize>>();
    permutation
}

pub fn random_permutation(data: &mut Vec<usize>, n: usize) {
    for i in (0..n).rev() {
        let j = rand::thread_rng().gen_range(0..=i);
        data.swap(i, j);
    }
}
