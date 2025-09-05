use core::f32;
use std::{
    io::{self, Write},
    iter::zip,
};

mod dataloader;
mod tokenizer;

// T = sequence_length
// B = batch_size
// C = channels
// V = vocab_size
// Vp = vocabulary padded (for efficiency)

fn matmul_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) -> Vec<f32> {
    // input (B,T,C)
    // output (B,T,OC)
    let mut output = Vec::with_capacity(B * T * OC);
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;
            for o in 0..OC {
                let mut val: f32;
                if let Some(b) = bias {
                    val = b[o];
                } else {
                    val = 0.0;
                };
                for i in 0..C {
                    val += input[bt * C + i] * weight[o * C + i];
                }
                output.push(val);
            }
        }
    }
    output
}

fn matmul_backward(
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // weight (OC, C)
    // bias (OC)
    // dout (B, T, OC)
    // inp (B, T, C)
    let mut dinp = vec![0f32; B * T * C];
    let mut dweight = vec![0f32; OC * C];
    let mut dbias = vec![0f32; OC];

    for b in 0..B {
        for t in 0..T {
            let btc = b * T * C + t * C;
            let btoc = b * T * OC + t * OC;

            for o in 0..OC {
                // gradient contribution to bias
                dbias[o] += dout[btoc + o];
                for c in 0..C {
                    // gradient contribution to weight
                    dweight[o * C + c] += inp[btc + c] * dout[btoc + o];
                    // gradient contribution to input
                    dinp[btc + c] += weight[o * C + c] * dout[btoc + o];
                }
            }
        }
    }
    (dinp, dweight, dbias)
}

fn gelu_forward(input: &[f32], N: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(N);
    let scaling_factor = f32::sqrt(2.0 / f32::consts::PI);
    for i in 0..N {
        let x = input[i];
        let cube = 0.44715 * x * x * x;
        output.push(0.5 * x * (1.0 + f32::tanh(scaling_factor * (x + cube))))
    }
    output
}

fn gelu_backward(dout: &[f32], input: &[f32], N: usize) -> Vec<f32> {
    let mut dinp = vec![0f32; N];
    let scaling_factor = f32::sqrt(2.0 / f32::consts::PI);
    for i in 0..N {
        let x = input[i];
        let cube = 0.44715 * x * x * x;
        let tanh_arg = scaling_factor * (x + cube);
        let tanh_out = f32::tanh(tanh_arg);
        let coshf_out = f32::cosh(tanh_arg);
        let sech_out = 1.0 / (coshf_out * coshf_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += local_grad * dout[i]
    }
    dinp
}

fn residual_forward(input1: &[f32], input2: &[f32], N: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        output.push(input1[i] + input2[i])
    }
    output
}

fn residual_backward(dout: &[f32], N: usize) -> (Vec<f32>, Vec<f32>) {
    let mut dinp1: Vec<f32> = Vec::with_capacity(N);
    let mut dinp2: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        dinp1.push(dout[i]);
        dinp2.push(dout[i]);
    }
    (dinp1, dinp2)
}

fn attention_forward(
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) -> Vec<f32> {
    // input (B,T,3C) holds (Q,K,V) values
    // preatt, att (B, NH, T, T)
    // NH = number of heads
    // T = sequence length
    // output (B,T,C)
    let mut output: Vec<f32> = vec![0.0; B * T * C];
    let C3 = 3 * C;
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_idx = b * T * C3 + t * C3 + h * hs;
                let preatt_base = b * NH * T * T + h * T * T + t * T;
                let att_base = b * NH * T * T + h * T * T + t * T; // att[b, h, t, :]

                // pass1: calculate query dot key and maxval
                let mut maxval: f32 = f32::MIN;

                for t2 in 0..t {
                    let mut val: f32 = 0.0;
                    let key_idx = b * T * C3 + t2 * C3 + h * hs + C; // + C because it's key

                    for i in 0..hs {
                        val += inp[query_idx + i] * inp[key_idx + i]
                    }
                    val *= scale;

                    if val > maxval {
                        maxval = val;
                    }

                    // preatt[b, h, t, t2]
                    preatt[preatt_base + t2] = val;
                }

                // pass2: calcul the exp and keep track of sum
                let mut expsum: f32 = 0.0;
                for t2 in 0..t {
                    let expv = (preatt[preatt_base + t2] - maxval).exp();
                    expsum += expv;
                    att[att_base + t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // pass3: normalize to get the softmax
                for t2 in 0..T {
                    if t2 <= t {
                        att[att_base + t2] *= expsum_inv;
                    } else {
                        att[att_base + t2] = 0.0;
                    }
                }
                // pass4: accumulate weighted values into the output of attention
                let out_base = b * T * C + t * C + h * hs; // output[b, t, h, :]
                for t2 in 0..t {
                    let value_idx = b * T * C3 + t2 * C3 + h * hs + 2 * C; // + 2*C because it's value
                    for i in 0..hs {
                        output[out_base + i] += att[att_base + t2] * inp[value_idx + i];
                    }
                }
            }
        }
    }
    output
}

fn attention_backward(
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // inp (B, T, 3C)
    // dout (B, T, C)
    // preatt, att (B, NH, T, T)
    let dinp = vec![0f32; B * T * C];
    let dpreatt = vec![0f32; B * NH * T * T];
    let datt = vec![0f32; B * NH * T * T];
    //todo: implement
    (dinp, dpreatt, datt)
}

fn encoder_forward(
    input: &[usize],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) -> Vec<f32> {
    // input (B,T)
    // output (B,T,C)
    // wte (V,C): weight token embedding
    // V is vocabulary size
    // wpe (maxT, C): weight positional embedding
    let mut output = Vec::with_capacity(B * T * C);
    for b in 0..B {
        for t in 0..T {
            // Get token_id
            let token_id: usize = input[b * T + t];
            for c in 0..C {
                output.push(wte[token_id * C + c] + wpe[t * C + c]);
            }
        }
    }
    output
}

fn encoder_backward(
    dout: &[f32],
    input: &[usize],
    B: usize,
    T: usize,
    C: usize,
    V: usize,
) -> (Vec<f32>, Vec<f32>) {
    // input (B, T)
    // dout (B,T,C)
    // wte (V,C): weight token embedding
    // V is vocabulary size
    // wpe (maxT, C): weight positional embedding
    let mut dwte = vec![0f32; V * C];
    let mut dwpe = vec![0f32; V * C];

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
    (dwte, dwpe)
}

fn softmax_forward(
    logits: &[f32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
    temperature: f32,
) -> Vec<f32> {
    // input (B,T,Vp)
    // output (B,T,Vp)
    // Vp is the padded vocab size
    // V is the real vocab size
    let mut probs: Vec<f32> = vec![0f32; B * T * Vp];
    for b in 0..B {
        for t in 0..T {
            let btv = b * T * Vp + t * Vp;
            let mut sum: f32 = 0.0;
            for v in 0..V {
                sum += f32::exp(logits[btv + v] / temperature);
            }
            for v in 0..V {
                probs[btv + v] = f32::exp(logits[btv + v] / temperature) / sum;
            }
        }
    }
    probs
}

fn crossentropy_forward(
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
            output.push(-probs[b * T * Vp + t * Vp + ix].log2())
        }
    }
    output
}

fn crossentropy_softmax_backward(
    dlosses: &[f32],
    targets: &[usize],
    probs: &[f32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) -> Vec<f32> {
    let mut dlogits: Vec<f32> = vec![0f32; B * T * Vp];
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
    dlogits
}

fn layernorm_forward(
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) -> Vec<f32> {
    // inp (B,T,C)
    // output (B,T,C)
    // mean and rstd are (B,T)
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    let mut output = Vec::new();
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
                output.push(o);
            }

            // cache mean and std for later
            mean[b * T + t] = m;
            rstd[b * T + t] = reciprocal_std;
        }
    }
    output
}

fn layernorm_backward(
    dout: &[f32],
    inp: &[f32],
    mean: &[f32],
    rstd: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // inp (B,T,C)
    // dout (B,T,C)
    // mean and rstd are (B,T)
    let mut dweight = vec![0f32; C];
    let mut dbias = vec![0f32; C];
    let mut dinp = vec![0f32; B * T * C];
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
                dinp[bt + c] = dval
            }
        }
    }
    (dinp, dweight, dbias)
}

pub struct ActivationTensors {
    encoded: Vec<f32>,   // (B, T, C)
    ln1: Vec<f32>,       // (L, B, T, C)
    ln1_mean: Vec<f32>,  // (L, B, T)
    ln1_rstd: Vec<f32>,  // (L, B, T)
    qkv: Vec<f32>,       // (L, B, T, 3*C)
    atty: Vec<f32>,      // (L, B, T, C)
    preatt: Vec<f32>,    // (L, B, NH, T, T)
    att: Vec<f32>,       // (L, B, NH, T, T)
    attproj: Vec<f32>,   // (L, B, T, C)
    residual2: Vec<f32>, // (L, B, T, C)
    ln2: Vec<f32>,       // (L, B, T, C)
    ln2_mean: Vec<f32>,  // (L, B, T)
    ln2_rstd: Vec<f32>,  // (L, B, T)
    fch: Vec<f32>,       // (L, B, T, 4*C)
    fch_gelu: Vec<f32>,  // (L, B, T, 4*C)
    fcproj: Vec<f32>,    // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    lnf: Vec<f32>,       // (B, T, C)
    lnf_mean: Vec<f32>,  // (B, T)
    lnf_rstd: Vec<f32>,  // (B, T)
    logits: Vec<f32>,    // (B, T, V)
    probs: Vec<f32>,     // (B, T, V)
    losses: Vec<f32>,    // (B, T)
}

impl IntoIterator for ActivationTensors {
    type Item = Vec<f32>;
    type IntoIter = std::array::IntoIter<Vec<f32>, 23>;

    fn into_iter(self) -> Self::IntoIter {
        [
            self.encoded,
            self.ln1,
            self.ln1_mean,
            self.ln1_rstd,
            self.qkv,
            self.atty,
            self.preatt,
            self.att,
            self.attproj,
            self.residual2,
            self.ln2,
            self.ln2_mean,
            self.ln2_rstd,
            self.fch,
            self.fch_gelu,
            self.fcproj,
            self.residual3,
            self.lnf,
            self.lnf_mean,
            self.lnf_rstd,
            self.logits,
            self.probs,
            self.losses,
        ]
        .into_iter()
    }
}

pub struct ParameterTensors {
    wte: Vec<f32>,      // (V, C)
    wpe: Vec<f32>,      // (maxT, C)
    ln1w: Vec<f32>,     // (L, C)
    ln1b: Vec<f32>,     // (L, C)
    qkvw: Vec<f32>,     // (L, 3*C, C)
    qkvb: Vec<f32>,     // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    ln2w: Vec<f32>,     // (L, C)
    ln2b: Vec<f32>,     // (L, C)
    fcw: Vec<f32>,      // (L, 4*C, C)
    fcb: Vec<f32>,      // (L, 4*C)
    fcprojw: Vec<f32>,  // (L, C, 4*C)
    fcprojb: Vec<f32>,  // (L, C)
    lnfw: Vec<f32>,     // (C)
    lnfb: Vec<f32>,     // (C)
}

// Allow iterating mutably over parameter tensors without moving them,
// yielding each parameter vector as a mutable reference for in-place updates.
impl<'a> IntoIterator for &'a mut ParameterTensors {
    type Item = &'a mut Vec<f32>;
    type IntoIter = std::array::IntoIter<&'a mut Vec<f32>, 16>;

    fn into_iter(self) -> Self::IntoIter {
        let ParameterTensors {
            wte,
            wpe,
            ln1w,
            ln1b,
            qkvw,
            qkvb,
            attprojw,
            attprojb,
            ln2w,
            ln2b,
            fcw,
            fcb,
            fcprojw,
            fcprojb,
            lnfw,
            lnfb,
        } = self;

        [
            wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw,
            fcprojb, lnfw, lnfb,
        ]
        .into_iter()
    }
}

pub struct Config {
    vocab_size: usize,
    padded_vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    channels: usize,
    max_seq_len: usize,
    version: u32,
    B: usize,
    T: usize,
}

fn init_params(sizes: [usize; 16]) -> ParameterTensors {
    ParameterTensors {
        wte: vec![0.0; sizes[0]],
        wpe: vec![0.0; sizes[1]],
        ln1w: vec![0.0; sizes[2]],
        ln1b: vec![0.0; sizes[3]],
        qkvw: vec![0.0; sizes[4]],
        qkvb: vec![0.0; sizes[5]],
        attprojw: vec![0.0; sizes[6]],
        attprojb: vec![0.0; sizes[7]],
        ln2w: vec![0.0; sizes[8]],
        ln2b: vec![0.0; sizes[9]],
        fcw: vec![0.0; sizes[10]],
        fcb: vec![0.0; sizes[11]],
        fcprojw: vec![0.0; sizes[12]],
        fcprojb: vec![0.0; sizes[13]],
        lnfw: vec![0.0; sizes[14]],
        lnfb: vec![0.0; sizes[15]],
    }
}

fn init_acts(sizes: [usize; 23]) -> ActivationTensors {
    ActivationTensors {
        encoded: vec![0.0; sizes[0]],
        ln1: vec![0.0; sizes[1]],
        ln1_mean: vec![0.0; sizes[2]],
        ln1_rstd: vec![0.0; sizes[3]],
        qkv: vec![0.0; sizes[4]],
        atty: vec![0.0; sizes[5]],
        preatt: vec![0.0; sizes[6]],
        att: vec![0.0; sizes[7]],
        attproj: vec![0.0; sizes[8]],
        residual2: vec![0.0; sizes[9]],
        ln2: vec![0.0; sizes[10]],
        ln2_mean: vec![0.0; sizes[11]],
        ln2_rstd: vec![0.0; sizes[12]],
        fch: vec![0.0; sizes[13]],
        fch_gelu: vec![0.0; sizes[14]],
        fcproj: vec![0.0; sizes[15]],
        residual3: vec![0.0; sizes[16]],
        lnf: vec![0.0; sizes[17]],
        lnf_mean: vec![0.0; sizes[18]],
        lnf_rstd: vec![0.0; sizes[19]],
        logits: vec![0.0; sizes[20]],
        probs: vec![0.0; sizes[21]],
        losses: vec![0.0; sizes[22]],
    }
}

fn load_model(file: &str, B: usize, T: usize) -> GPT2 {
    // Load model
    let bytes = std::fs::read(file).unwrap();

    let mut model_header: [u32; 256] = [0; 256];
    let mut iter = bytes.chunks_exact(4);
    for i in 0..256 {
        let byte_4 = iter.next().unwrap();
        let double = u32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        model_header[i] = double;
    }
    // println!("Model header {:?}", model_header);

    // Init config
    let config = Config {
        version: model_header[1],
        max_seq_len: model_header[2] as usize,
        vocab_size: model_header[3] as usize,
        num_layers: model_header[4] as usize,
        num_heads: model_header[5] as usize,
        channels: model_header[6] as usize,
        padded_vocab_size: model_header[7] as usize,
        B: B,
        T: T,
    };

    println!("Vocabulary size: {}", config.vocab_size);
    println!("Padded vocabulary size: {}", config.padded_vocab_size);

    let mut param_sizes = [0; 16];

    let Vp = config.padded_vocab_size;
    let C = config.channels;
    let maxT = config.max_seq_len;
    let L = config.num_layers;

    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    // println!("{:?}", param_sizes);

    // Total parameters
    let mut num_parameters: usize = 0;
    for n_params in param_sizes {
        num_parameters += n_params;
    }
    println!(
        "Total parameters: {}M",
        num_parameters as u32 / 10u32.pow(6)
    );

    let mut matrices = Vec::new();
    for size in param_sizes.iter() {
        let mut i: usize = 0;
        let mut matrix: Vec<f32> = vec![0f32; *size];
        while i < *size {
            let byte_4 = iter.next().unwrap();
            let double = f32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
            matrix[i] = double;
            i += 1;
        }
        matrices.push(matrix);
        // println!("{}", i);
    }

    let params = ParameterTensors {
        wte: matrices[0].clone(),
        wpe: matrices[1].clone(),
        ln1w: matrices[2].clone(),
        ln1b: matrices[3].clone(),
        qkvw: matrices[4].clone(),
        qkvb: matrices[5].clone(),
        attprojw: matrices[6].clone(),
        attprojb: matrices[7].clone(),
        ln2w: matrices[8].clone(),
        ln2b: matrices[9].clone(),
        fcw: matrices[10].clone(),
        fcb: matrices[11].clone(),
        fcprojw: matrices[12].clone(),
        fcprojb: matrices[13].clone(),
        lnfw: matrices[14].clone(),
        lnfb: matrices[15].clone(),
    };
    drop(matrices);

    let (num_activations, act_sizes) = init_activations(&config);

    let acts = ActivationTensors {
        encoded: vec![0.0; act_sizes[0]],
        ln1: vec![0.0; act_sizes[1]],
        ln1_mean: vec![0.0; act_sizes[2]],
        ln1_rstd: vec![0.0; act_sizes[3]],
        qkv: vec![0.0; act_sizes[4]],
        atty: vec![0.0; act_sizes[5]],
        preatt: vec![0.0; act_sizes[6]],
        att: vec![0.0; act_sizes[7]],
        attproj: vec![0.0; act_sizes[8]],
        residual2: vec![0.0; act_sizes[9]],
        ln2: vec![0.0; act_sizes[10]],
        ln2_mean: vec![0.0; act_sizes[11]],
        ln2_rstd: vec![0.0; act_sizes[12]],
        fch: vec![0.0; act_sizes[13]],
        fch_gelu: vec![0.0; act_sizes[14]],
        fcproj: vec![0.0; act_sizes[15]],
        residual3: vec![0.0; act_sizes[16]],
        lnf: vec![0.0; act_sizes[17]],
        lnf_mean: vec![0.0; act_sizes[18]],
        lnf_rstd: vec![0.0; act_sizes[19]],
        logits: vec![0.0; act_sizes[20]],
        probs: vec![0.0; act_sizes[21]],
        losses: vec![0.0; act_sizes[22]],
    };

    let gpt2 = GPT2 {
        num_parameters,
        num_activations,
        header: model_header,
        config,
        params,
        param_sizes,
        acts,
        act_sizes,
        grads_acts: init_acts(act_sizes),
        grads: init_params(param_sizes),
        m_memory: vec![0f32; num_parameters],
        v_memory: vec![0f32; num_parameters],
    };

    gpt2
}

fn init_activations(config: &Config) -> (usize, [usize; 23]) {
    // Activations
    let mut act_sizes = [0; 23];
    let B: usize = config.B; // batch size
    let T: usize = config.T; // seq_len
    let C: usize = config.channels;
    let NH: usize = config.num_heads;
    let L: usize = config.num_layers;
    let Vp: usize = config.padded_vocab_size;

    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses

    let mut num_activations = 0;
    for size in act_sizes {
        num_activations += size;
    }
    println!("Total activations: {}M", num_activations / 10usize.pow(6));

    (num_activations, act_sizes)
}

pub struct GPT2 {
    num_parameters: usize,
    num_activations: usize,
    header: [u32; 256],
    config: Config,
    params: ParameterTensors,
    param_sizes: [usize; 16],
    acts: ActivationTensors,
    act_sizes: [usize; 23],
    // gradients
    grads: ParameterTensors,
    grads_acts: ActivationTensors,
    // memory for adamw
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
}

impl GPT2 {
    fn forward(&mut self, inputs: &[usize], temperature: f32) -> Vec<f32> {
        let B: usize = self.config.B;
        let T: usize = self.config.T;
        let V: usize = self.config.vocab_size;
        let Vp: usize = self.config.padded_vocab_size;
        let L: usize = self.config.num_layers;
        let NH: usize = self.config.num_heads;
        let C: usize = self.config.channels;

        let acts = &mut self.acts;
        let params = &mut self.params;

        // forward pass
        let mut residual: &[f32];
        acts.encoded = encoder_forward(inputs, &params.wte, &params.wpe, B, T, C);
        for l in 0..L {
            // println!("Layer n°{}", l);
            if l == 0 {
                residual = &acts.encoded; // (B, T, C)
            } else {
                residual = &acts.residual3[l * B * T * C..]; // (L, B, T, C)
            }
            acts.ln1[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&layernorm_forward(
                &mut acts.ln1_mean[l * B * T..],
                &mut acts.ln1_rstd[l * B * T..],
                residual,
                &params.ln1w[l * C..],
                &params.ln1b[l * C..],
                B,
                T,
                C,
            ));
            acts.qkv[l * B * T * 3 * C..(l + 1) * B * T * 3 * C].copy_from_slice(&matmul_forward(
                &acts.ln1[l * B * T * C..],
                &params.qkvw[l * 3 * C * C..],
                Some(&params.qkvb[l * 3 * C..]),
                B,
                T,
                C,
                3 * C,
            ));
            acts.atty[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&attention_forward(
                &mut acts.preatt[l * B * NH * T * T..],
                &mut acts.att[l * B * NH * T * T..],
                &acts.qkv[l * B * T * 3 * C..],
                B,
                T,
                C,
                NH,
            ));
            acts.attproj[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&matmul_forward(
                &acts.atty[l * B * T * C..],
                &params.attprojw[l * C * C..],
                Some(&params.attprojb[l * C..]),
                B,
                T,
                C,
                C,
            ));
            acts.residual2[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&residual_forward(
                &residual,
                &acts.attproj[l * B * T * C..],
                B * T * C,
            ));
            acts.ln2[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&layernorm_forward(
                &mut acts.ln2_mean[l * B * T..],
                &mut acts.ln2_rstd[l * B * T..],
                &acts.residual2[l * B * T * C..],
                &params.ln2w[l * C..],
                &params.ln2b[l * C..],
                B,
                T,
                C,
            ));
            acts.fch[l * B * T * 4 * C..(l + 1) * B * T * 4 * C].copy_from_slice(&matmul_forward(
                &acts.ln2[l * B * T * C..],
                &params.fcw[l * 4 * C * C..],
                Some(&params.fcb[l * 4 * C..]),
                B,
                T,
                C,
                4 * C,
            ));
            acts.fch_gelu[l * B * T * 4 * C..(l + 1) * B * T * 4 * C]
                .copy_from_slice(&gelu_forward(&acts.fch[l * B * T * 4 * C..], B * T * 4 * C));
            acts.fcproj[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&matmul_forward(
                &acts.fch_gelu[l * B * T * 4 * C..],
                &params.fcprojw[l * 4 * C * C..],
                Some(&params.fcprojb[l * C..]),
                B,
                T,
                4 * C,
                C,
            ));
            acts.residual3[l * B * T * C..(l + 1) * B * T * C].copy_from_slice(&residual_forward(
                &acts.residual2[l * B * T * C..],
                &acts.fcproj[l * B * T * C..],
                B * T * C,
            ));
        }
        // println!("Layers applied");
        acts.lnf = layernorm_forward(
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            &acts.residual3[(L - 1) * B * T * C..],
            &params.lnfw,
            &params.lnfb,
            B,
            T,
            C,
        );
        acts.logits = matmul_forward(&acts.lnf, &params.wte, None, B, T, C, Vp);
        acts.probs = softmax_forward(&acts.logits, B, T, V, Vp, temperature);
        // output is (B,T,Vp)
        return acts.probs.clone();
    }

    fn backward(&mut self, inputs: &[usize], targets: &[usize]) {
        let B: usize = self.config.B;
        let T: usize = self.config.T;
        let V: usize = self.config.vocab_size;
        let Vp: usize = self.config.padded_vocab_size;
        let L: usize = self.config.num_layers;
        let NH: usize = self.config.num_heads;
        let C: usize = self.config.channels;

        let acts = &mut self.acts;
        let params = &mut self.params;
        let grads_acts = &mut self.grads_acts;
        let grads = &mut self.grads;

        // Fill with the mean loss
        let dloss_mean = 1.0 / (B * T) as f32;
        for i in 0..B * T {
            grads_acts.losses[i] = dloss_mean;
        }

        grads_acts.logits =
            crossentropy_softmax_backward(&grads_acts.losses, targets, &acts.probs, B, T, V, Vp);
        (grads_acts.lnf, grads.wte, _) =
            matmul_backward(&grads_acts.logits, &acts.lnf, &params.wte, B, T, C, Vp);
        (grads_acts.residual3, grads.lnfw, grads.lnfb) = layernorm_backward(
            &grads_acts.lnf,
            &acts.residual3,
            &acts.lnf_mean,
            &acts.lnf_rstd,
            &params.lnfw,
            B,
            T,
            C,
        );
        let mut residual: &[f32];
        let mut grads_residual: Vec<f32>;
        for l in 0..L {
            (grads_acts.residual2, grads_acts.fcproj) =
                residual_backward(&grads_acts.residual3, B * T * C);
            (grads_acts.fch_gelu, grads.fcprojw, grads.fcprojb) = matmul_backward(
                &grads_acts.fcproj,
                &acts.fch_gelu,
                &params.fcprojw,
                B,
                T,
                4 * C,
                C,
            );
            grads_acts.fch = gelu_backward(&grads_acts.fch_gelu, &acts.fch, B * T * 4 * C);
            (grads_acts.ln2, grads.fcw, grads.fcb) =
                matmul_backward(&grads_acts.fch, &acts.ln2, &params.fcw, B, T, C, 4 * C);
            (grads_acts.residual2, grads.ln2w, grads.ln2b) = layernorm_backward(
                &grads_acts.ln2,
                &acts.residual2,
                &acts.ln2_mean,
                &acts.ln2_rstd,
                &params.ln2w,
                B,
                T,
                C,
            );
            (_, grads_acts.attproj) = residual_backward(&grads_acts.residual2, B * T * C);
            (grads_acts.atty, grads.attprojw, grads.attprojb) = matmul_backward(
                &grads_acts.attproj,
                &acts.atty,
                &params.attprojw,
                B,
                T,
                C,
                C,
            );
            (grads_acts.atty, grads_acts.preatt, grads_acts.att) =
                attention_backward(&grads_acts.atty, &acts.att, &acts.qkv, B, T, C, NH);
            (grads_acts.ln1, grads.qkvw, _) =
                matmul_backward(&grads_acts.qkv, &acts.ln1, &params.qkvw, B, T, C, 3 * C);

            if l == 0 {
                residual = &acts.encoded;
            } else {
                residual = &acts.residual3;
            }

            (grads_residual, grads.ln1w, grads.ln1b) = layernorm_backward(
                &grads_acts.ln1,
                residual,
                &acts.ln1_mean,
                &acts.ln1_rstd,
                &params.ln1w,
                B,
                T,
                C,
            );

            if l == 0 {
                grads_acts.encoded = grads_residual;
            } else {
                grads_acts.residual3 = grads_residual;
            }
        }
        (grads.wte, grads.wpe) = encoder_backward(&grads_acts.encoded, inputs, B, T, C, V);
    }

    fn loss(&mut self, probs: &[f32], targets: &[usize]) -> f32 {
        // targets is (B,T)
        let B: usize = self.config.B;
        let Vp: usize = self.config.padded_vocab_size;
        let V: usize = self.config.vocab_size;
        let T: usize = self.config.T;
        let acts = &mut self.acts;
        acts.losses = crossentropy_forward(&probs, targets, B, T, Vp);
        let mut mean_loss: f32 = 0.0;
        for loss in &acts.losses {
            mean_loss += loss;
        }
        mean_loss /= (B * T) as f32;
        println!("Mean loss: {}", mean_loss);

        // Check the probability that the model is giving to the target token
        // It should increase during training
        if true {
            for (i, target_token) in targets.iter().enumerate() {
                let target_token_probability = probs[i * Vp + target_token];
                println!(
                    "Expected token {} - Probability {}",
                    target_token, target_token_probability
                );
            }
        }

        mean_loss
    }

    fn argmax(&self, probs: Vec<f32>, only_last: bool) -> Vec<usize> {
        // output is (B, T) if only_last=false
        // (B) if only_last=true
        // get the token with the highest probability for each dimension (temperature=1)
        // println!("Sampling token with the highest probability");
        let B = self.config.B;
        let T = self.config.T;
        let Vp = self.config.padded_vocab_size;
        let mut token_ids: Vec<usize>;
        if only_last {
            // check only the last
            token_ids = vec![0; B];
            for b in 0..B {
                let start_idx = b * T * Vp + (T - 1) * Vp;
                let end_idx = (b + 1) * T * Vp;
                let token_id = probs[start_idx..end_idx]
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.partial_cmp(y).expect("Partial comparison paniced"))
                    .map(|(index, _)| index)
                    .expect("No index was found");
                token_ids[b] = token_id;
            }
        } else {
            token_ids = vec![0; B * T];
            // find the most probable token for each b,t
            for b in 0..B {
                for t in 0..T {
                    let start_idx = b * T * Vp + t * Vp;
                    let end_idx = b * T * Vp + (t + 1) * Vp;
                    let token_id = probs[start_idx..end_idx]
                        .iter()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| {
                            x.partial_cmp(y).expect("Partial comparison paniced")
                        })
                        .map(|(index, _)| index)
                        .expect("No index was found");
                    token_ids[b * T + t] = token_id;
                }
            }
        }
        token_ids
    }

    fn sample(
        &mut self,
        inputs: Vec<usize>,
        N: usize,
        temperature: f32,
        tokenizer: &tokenizer::Tokenizer,
    ) -> Vec<usize> {
        // inputs (B, T)
        // outputs (B, T + N)
        let B = self.config.B;
        let T: usize = self.config.T;
        // println!("Batch sampling of {} tokens", N);
        let mut modified_inputs = inputs.clone();
        let mut outputs = inputs.clone();
        for i in 0..N {
            // println!("Generate token n°{}", i);
            // inputs is (1,T)
            // println!("Inputs {:?}", modified_inputs);

            // probs is (B, T, Vp)
            let probs: Vec<f32> = self.forward(&modified_inputs, temperature);

            // (B) due to only_last=true
            let tokens = self.argmax(probs, true);

            for b in 0..B {
                // Insert at the end of the batched input
                outputs.insert((b + 1) * (T + i), tokens[b]);
                // Insert at the end of the sentence
                modified_inputs.insert((b + 1) * T, tokens[b]);
                // Remove the first token
                modified_inputs.remove(b * T);

                if b == 0 {
                    // println!("Sampled token {}", tokens[b]);
                    let token = tokenizer.decode(tokens[b]);
                    print!("{}", token);
                    io::stdout().flush().unwrap(); // Force flush
                }
            }
        }
        outputs
    }

    fn generate(
        &mut self,
        inputs: Vec<usize>,
        N: usize,
        temperature: f32,
        tokenizer: &tokenizer::Tokenizer,
    ) {
        let T = self.config.T;
        //TODO: ignore B actually
        let input_text = tokenizer.batch_decode(&inputs[0..T]);
        println!("Input text: {}", input_text);
        self.sample(inputs, N, temperature, tokenizer);
    }

    fn zero_grad(&mut self) {
        self.grads_acts = init_acts(self.act_sizes);
        self.grads = init_params(self.param_sizes);
    }

    fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: usize,
    ) {
        // implement adamw algorithm
        let GPT2 {
            params,
            grads,
            m_memory,
            v_memory,
            ..
        } = self;
        let mut i: usize = 0;

        for (param_vec, grad_vec) in zip(params, grads) {
            for (param, grad) in zip(param_vec.iter_mut(), grad_vec.iter()) {
                // Update biased first moment estimate
                let m: f32 = beta1 * m_memory[i] + (1.0 - beta1) * *grad;
                // Update biased second raw moment estimate
                let v: f32 = beta2 * v_memory[i] + (1.0 - beta2) * (*grad * *grad);
                // Compute bias-corrected estimates (simple form)
                let m_hat: f32 = m / (1.0 - beta1);
                // Compute bias-corrected second raw moment estimate
                let v_hat: f32 = v / (1.0 - beta2);

                // Persist optimizer state
                m_memory[i] = m;
                v_memory[i] = v;

                // AdamW parameter update
                *param -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *param);

                i += 1;
            }
        }
    }
}

fn main() {
    let B: usize = 1;
    let T: usize = 64;
    let temperature: f32 = 1.0;
    let tokens_to_sample: usize = 10;

    let train_data_path =
        "/Users/gphilippe/dev/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    let val_data_path =
        "/Users/gphilippe/dev/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    let model_path = "/Users/gphilippe/dev/llm.c/gpt2_124M.bin";
    // let model_path = "/Users/gphilippe/dev/llm.c/gpt2_124M_debug_state.bin";
    let tokenizer_path = "/Users/gphilippe/dev/llm.c/gpt2_tokenizer.bin";

    let tokenizer = tokenizer::load_tokenizer(&tokenizer_path);
    // Check that tokens
    // Fix with new dataloader
    // tokenizer.check(&dataset);

    // Build inputs and target
    // Avoid to load everything directly
    // println!("Dataset length: {}", inputs.len());

    let mut train_dataloader = dataloader::create_dataloader(train_data_path, B, T);
    println!("Load train dataloader with {} tokens", train_dataloader.len);
    let mut val_dataloader = dataloader::create_dataloader(val_data_path, B, T);
    println!("Load val dataloader with {} tokens", train_dataloader.len);

    // Decode the dataset
    // let text = tokenizer.batch_decode(&targets);
    // println!("Text {}", text);

    let mut gpt = load_model(&model_path, B, T);

    println!("Max sequence length: {}", gpt.config.max_seq_len);

    // Check input data
    // let inputs_batch = &inputs[0..B*T];
    // println!("{:?}", &inputs_batch);
    // let text = tokenizer.batch_decode(&inputs_batch);
    // println!("Text is {}", text);
    // let targets_batch = targets[0..B*T];

    // Sample N tokens
    train_dataloader.next_batch();
    let batch_inputs = train_dataloader.inputs[0..B * T].to_vec();
    gpt.generate(batch_inputs, tokens_to_sample, temperature, &tokenizer);

    // todo: implement the train loop
    // how inputs are batched ?
    // implement backward methods
    // compare loss at first iteration

    // for step in 0..40 {
    //     // get data batch
    //     train_dataloader.next_batch();
    //     // forward pass
    //     println!("Inputs len: {}", &train_dataloader.inputs.len());
    //     let probs: Vec<f32> = gpt.forward(&train_dataloader.inputs, temperature);
    //     let loss: f32 = gpt.loss(&probs, &train_dataloader.targets);
    //     println!("Loss {} at step {}", loss, step);
    //     // zero grad
    //     gpt.zero_grad();
    //     // backward pass
    //     gpt.backward(&train_dataloader.inputs, &train_dataloader.targets);
    //     // update
    //     gpt.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
    // }
}
