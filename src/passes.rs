use core::f32;
use rayon::prelude::*;

// T = sequence_length
// B = batch_size
// C = channels
// V = vocab_size
// Vp = vocabulary padded (for efficiency)

pub fn matmul_forward_naive(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    _B: usize,
    _T: usize,
    C: usize,
    OC: usize,
) {
    // input (B,T,C)
    // output (B,T,OC)
    output
        .par_chunks_mut(OC)
        .enumerate()
        .for_each(|(bt, out_row)| {
            let inp_row = &input[bt * C..(bt + 1) * C];
            for o in 0..OC {
                let mut acc: f32 = if let Some(b) = bias.as_ref() {
                    b[o]
                } else {
                    0.0
                };
                let w_row = &weight[o * C..(o + 1) * C];
                // dot product inp_row (C) · w_row (C)
                // manual loop to avoid iterator overhead in hot path
                for i in 0..C {
                    acc += inp_row[i] * w_row[i];
                }
                out_row[o] = acc;
            }
        });
}

pub fn matmul_forward(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    unsafe {
        matrixmultiply::sgemm(
            B * T,
            C,
            OC,
            1.0,
            input.as_ptr(),
            C as isize,
            1,
            weight.as_ptr(),
            1,
            C as isize,
            0.0,
            output.as_mut_ptr(),
            OC as isize,
            1,
        )
    };

    if let Some(b) = bias.as_deref() {
        for bt in 0..B * T {
            for o in 0..OC {
                output[bt * OC + o] += b[o];
            }
        }
    }
}

pub fn matmul_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    mut dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // dout (B, T, OC)
    // inp (B, T, C)
    // weight (OC, C)

    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    unsafe {
        // dinp += dout @ weight
        matrixmultiply::sgemm(
            B * T,
            OC,
            C,
            1.0,
            dout.as_ptr(),
            OC as isize,
            1,
            weight.as_ptr(),
            C as isize,
            1,
            1.0,
            dinp.as_mut_ptr(),
            C as isize,
            1,
        );

        // dweight += dout^T @ inp
        matrixmultiply::sgemm(
            OC,
            B * T,
            C,
            1.0,
            dout.as_ptr(),
            1,
            OC as isize,
            inp.as_ptr(),
            C as isize,
            1,
            1.0,
            dweight.as_mut_ptr(),
            C as isize,
            1,
        );
    }

    // backward into bias: compute separately to avoid mutable aliasing across threads
    if let Some(db) = dbias.as_deref_mut() {
        for o in 0..OC {
            let mut sum = 0.0f32;
            for b in 0..B {
                for t in 0..T {
                    let btoc = b * T * OC + t * OC;
                    sum += dout[btoc + o];
                }
            }
            db[o] += sum;
        }
    }
}

pub fn gelu_forward(output: &mut [f32], input: &[f32], N: usize) {
    let scaling_factor = f32::sqrt(2.0 / f32::consts::PI);
    for i in 0..N {
        let x = input[i];
        let cube = 0.044715 * x * x * x;
        output[i] = 0.5 * x * (1.0 + f32::tanh(scaling_factor * (x + cube)));
    }
}

pub fn gelu_backward(dinp: &mut [f32], dout: &[f32], input: &[f32], N: usize) {
    let scaling_factor = f32::sqrt(2.0 / f32::consts::PI);
    for i in 0..N {
        let x = input[i];
        let cube = 0.044715 * x * x * x;
        let tanh_arg = scaling_factor * (x + cube);
        let tanh_out = f32::tanh(tanh_arg);
        let coshf_out = f32::cosh(tanh_arg);
        let sech_out = 1.0 / (coshf_out * coshf_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += local_grad * dout[i];
    }
}

pub fn residual_forward(output: &mut [f32], input1: &[f32], input2: &[f32], N: usize) {
    for i in 0..N {
        output[i] = input1[i] + input2[i];
    }
}

pub fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32], N: usize) {
    for i in 0..N {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

pub fn attention_forward(
    output: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    _B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // input (B,T,3C) holds (Q,K,V) values
    // preatt, att (B, NH, T, T)
    // NH = number of heads
    // T = sequence length
    // output (B,T,C)
    let C3 = 3 * C;
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    // parallelize over batch dimension with disjoint mutable chunks
    output
        .par_chunks_mut(T * C)
        .zip(preatt.par_chunks_mut(NH * T * T))
        .zip(att.par_chunks_mut(NH * T * T))
        .enumerate()
        .for_each(|(b, ((output_b, preatt_b), att_b))| {
            // Cache-friendly access pattern: process heads in order
            for h in 0..NH {
                for t in 0..T {
                    let query_idx = b * T * C3 + t * C3 + h * hs;
                    let preatt_base = h * T * T + t * T; // within this batch chunk
                    let att_base = h * T * T + t * T; // att[b, h, t, :]

                    // pass1: calculate query dot key and maxval
                    let mut maxval: f32 = f32::MIN;

                    for t2 in 0..=t {
                        let mut val: f32 = 0.0;
                        let key_idx = b * T * C3 + t2 * C3 + h * hs + C; // + C because it's key

                        for i in 0..hs {
                            val += inp[query_idx + i] * inp[key_idx + i];
                        }
                        val *= scale;

                        if val > maxval {
                            maxval = val;
                        }

                        // preatt[b, h, t, t2]
                        preatt_b[preatt_base + t2] = val;
                    }

                    // pass2: calcul the exp and keep track of sum
                    let mut expsum: f32 = 0.0;
                    for t2 in 0..=t {
                        let expv = (preatt_b[preatt_base + t2] - maxval).exp();
                        expsum += expv;
                        att_b[att_base + t2] = expv;
                    }
                    let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                    // pass3: normalize to get the softmax
                    for t2 in 0..T {
                        if t2 <= t {
                            att_b[att_base + t2] *= expsum_inv;
                        } else {
                            att_b[att_base + t2] = 0.0;
                        }
                    }
                    // pass4: accumulate weighted values into the output of attention
                    let out_base = t * C + h * hs; // within this batch chunk, output[b, t, h, :]
                    for i in 0..hs {
                        output_b[out_base + i] = 0.0;
                    }
                    for t2 in 0..=t {
                        let value_idx = b * T * C3 + t2 * C3 + h * hs + 2 * C; // + 2*C because it's value
                        let att_t2 = att_b[att_base + t2];
                        for i in 0..hs {
                            output_b[out_base + i] += att_t2 * inp[value_idx + i];
                        }
                    }
                }
            }
        });
}

pub fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // dout (B, T, C)
    // inp (B, T, 3C)
    // preatt, att (B, NH, T, T)
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1f32 / f32::sqrt(hs as f32);

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_idx = b * NH * T * T + h * T * T + t * T;
                let dout_idx = b * T * C + t * C + h * hs;
                let query_t_idx = b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                for t2 in 0..=t {
                    let value_idx = b * T * C3 + t2 * C3 + C * 2 + h * hs;
                    for i in 0..hs {
                        datt[att_idx + t2] += dout[dout_idx + i] * inp[value_idx + i];
                        dinp[value_idx + i] += dout[dout_idx + i] * att[att_idx + t2];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1f32 } else { 0f32 };
                        let local_derivate = att[att_idx + t2] * (indicator - att[att_idx + t3]);
                        dpreatt[att_idx + t3] += local_derivate * datt[att_idx + t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for t2 in 0..=t {
                    let key_t2_idx = b * T * C3 + t2 * C3 + C + h * hs;
                    for i in 0..hs {
                        dinp[query_t_idx + i] +=
                            inp[key_t2_idx + i] * scale * dpreatt[att_idx + t2];
                        dinp[key_t2_idx + i] +=
                            inp[query_t_idx + i] * scale * dpreatt[att_idx + t2];
                    }
                }
            }
        }
    }
}

pub fn encoder_forward(
    encoded: &mut [f32],
    input: &[usize],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // input (B,T)
    // output (B,T,C)
    // wte (V,C): weight token embedding
    // V is vocabulary size
    // wpe (maxT, C): weight positional embedding
    for b in 0..B {
        for t in 0..T {
            // Get token_id
            let token_id: usize = input[b * T + t];
            let ix = b * T * C + t * C;
            for c in 0..C {
                encoded[ix + c] = wte[token_id * C + c] + wpe[t * C + c];
            }
        }
    }
}

pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    input: &[usize],
    B: usize,
    T: usize,
    C: usize,
) {
    // input (B, T)
    // dout (B,T,C)
    // wte (V,C): weight token embedding
    // V is vocabulary size
    // wpe (maxT, C): weight positional embedding

    for b in 0..B {
        for t in 0..T {
            let token_id = input[b * T + t];
            let idx = b * T * C + t * C;
            for c in 0..C {
                let d = dout[idx + c];
                dwte[token_id * C + c] += d;
                dwpe[t * C + c] += d;
            }
        }
    }
}

pub fn softmax_forward(
    probs: &mut [f32],
    logits: &[f32],
    _B: usize,
    _T: usize,
    V: usize,
    Vp: usize,
    temperature: f32,
) {
    // logits (B,T,Vp)
    // probs (B,T,Vp)
    // Vp is the padded vocab size
    // V is the real vocab size
    probs
        .par_chunks_mut(Vp)
        .enumerate()
        .for_each(|(bt, probs_row)| {
            let btv = bt * Vp;

            // maxval is only calculated and subtracted for numerical stability
            let mut maxval = f32::MIN; // TODO something better
            for i in 0..V {
                if logits[btv + i] > maxval {
                    maxval = logits[btv + i];
                }
            }

            let mut sum: f32 = 0.0;
            for v in 0..V {
                probs_row[v] = f32::exp((logits[btv + v] - maxval) / temperature);
                sum += probs_row[v];
            }
            for v in 0..V {
                probs_row[v] /= sum;
            }
        });
}

pub fn crossentropy_forward(
    probs: &[f32],
    targets: &[usize],
    B: usize,
    T: usize,
    Vp: usize,
) -> Vec<f32> {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    let mut output: Vec<f32> = Vec::with_capacity(B * T);
    for b in 0..B {
        for t in 0..T {
            let ix: usize = targets[b * T + t];
            output.push(-probs[b * T * Vp + t * Vp + ix].ln())
        }
    }
    output
}

pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    targets: &[usize],
    probs: &[f32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let idx = b * T * Vp + t * Vp;
            let ix = targets[b * T + t];
            // let padded vocab gradient to 0
            for i in 0..V {
                let p = probs[idx + i];
                let indicator = if i == ix { 1.0 } else { 0.0 };
                dlogits[idx + i] += (p - indicator) * dlosses[b * T + t];
            }
        }
    }
}

pub fn layernorm_forward(
    output: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // inp (B,T,C)
    // output (B,T,C)
    // mean and rstd are (B,T)
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    let eps: f32 = 1e-5;
    for b in 0..B {
        for t in 0..T {
            let idx = b * T * C + t * C;
            let input_channels = &inp[idx..idx + C];
            // compute the mean
            let mut m: f32 = 0.0;
            for c in input_channels {
                m += c;
            }
            m = m / C as f32;

            // compute the variance
            let mut var: f32 = 0.0;
            for c in input_channels {
                let diff: f32 = c - m;
                var += diff * diff;
            }
            var = var / C as f32;

            // compute the rstd (reciprocal standard deviation)
            let reciprocal_std: f32 = 1.0 / (var + eps).sqrt();

            for i in 0..C {
                // normalize
                let n = reciprocal_std * (inp[idx + i] - m);
                // rescale
                let o = n * weight[i] + bias[i];
                output[idx + i] = o;
            }

            // cache mean and std for later
            mean[b * T + t] = m;
            rstd[b * T + t] = reciprocal_std;
        }
    }
}

pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    mean: &[f32],
    rstd: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    // inp (B,T,C)
    // dout (B,T,C)
    // mean and rstd are (B,T)
    for b in 0..B {
        for t in 0..T {
            let m = mean[b * T + t];
            let reciprocal_std = rstd[b * T + t];
            let bt = b * T * C + t * C;

            let mut dnorm_mean = 0f32;
            let mut dnorm_norm_mean = 0f32;
            for c in 0..C {
                let norm_bti = (inp[bt + c] - m) * reciprocal_std;
                let dnorm_i = weight[c] * dout[bt + c];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }

            dnorm_mean = dnorm_mean / C as f32;
            dnorm_norm_mean = dnorm_norm_mean / C as f32;

            for c in 0..C {
                let n = reciprocal_std * (inp[bt + c] - m);
                // gradient contribution to bias
                dbias[c] += dout[bt + c];
                // gradient contribution to weight
                dweight[c] += n * dout[bt + c];
                // gradient contribution to input
                let mut dval = 0f32;
                dval += weight[c] * dout[bt + c]; // term 1
                dval -= dnorm_mean; // term 2
                dval -= n * dnorm_norm_mean; // term 3
                dval *= reciprocal_std; // final scale
                dinp[bt + c] += dval;
            }
        }
    }
}
