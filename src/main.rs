#![allow(non_snake_case)]

use core::f32;
use std::{iter::zip, time::SystemTime};

mod dataloader;
mod passes;
mod tokenizer;
mod utils;
use passes::*;

// T = sequence_length
// B = batch_size
// C = channels
// V = vocab_size
// Vp = vocabulary padded (for efficiency)
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
    temperature: f32,
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

fn load_model(file: &str, temperature: f32) -> GPT2 {
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
        temperature: temperature,
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

    // Total parameters
    let mut num_parameters: usize = 0;
    for n_params in param_sizes {
        num_parameters += n_params;
    }
    println!(
        "Total parameters: {}M",
        num_parameters as u32 / 10u32.pow(6)
    );

    let params = load_params(param_sizes, iter);

    let gpt2 = GPT2 {
        num_parameters,
        num_activations: None,
        header: model_header,
        config,
        params,
        param_sizes,
        acts: None,
        act_sizes: None,
        grads_acts: None,
        grads: init_params(param_sizes),
        m_memory: vec![0f32; num_parameters],
        v_memory: vec![0f32; num_parameters],
    };

    gpt2
}

fn load_params(sizes: [usize; 16], mut iter: std::slice::ChunksExact<'_, u8>) -> ParameterTensors {
    let mut matrices = Vec::new();
    for size in sizes.iter() {
        let mut i: usize = 0;
        let mut matrix: Vec<f32> = vec![0f32; *size];
        while i < *size {
            let byte_4 = iter.next().unwrap();
            let double = f32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
            matrix[i] = double;
            i += 1;
        }
        matrices.push(matrix);
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
    params
}

fn load_acts(gpt: &mut GPT2, B: usize, T: usize) {
    let config = &gpt.config;
    let (num_activations, act_sizes) = init_activations(config, B, T);

    let acts = init_acts(act_sizes);

    gpt.num_activations = Some(num_activations);
    gpt.acts = Some(acts);
    gpt.act_sizes = Some(act_sizes);
}

fn load_debug_state(
    file: &str,
    gpt: &GPT2,
) -> (
    usize,
    usize,
    Vec<usize>,
    Vec<usize>,
    Vec<f32>,
    f32,
    ParameterTensors,
) {
    let bytes = std::fs::read(file).unwrap();
    let mut state_header: [u32; 256] = [0; 256];

    let mut iter = bytes.chunks_exact(4);
    for i in 0..256 {
        let byte_4 = iter.next().unwrap();
        let double = u32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        state_header[i] = double;
    }

    let B = state_header[2] as usize; // batch size, e.g. 4
    let T = state_header[3] as usize;
    let V = gpt.config.vocab_size;

    println!("[State]");
    println!("batch_size: {}", B);
    println!("seq_len: {}", T);

    let mut x: Vec<usize> = vec![0; B * T];
    for i in 0..B * T {
        let byte_4 = iter.next().unwrap();
        let double = u32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        x[i] = double as usize;
    }

    let mut y: Vec<usize> = vec![0; B * T];
    for i in 0..B * T {
        let byte_4 = iter.next().unwrap();
        let double = u32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        y[i] = double as usize;
    }

    let mut expected_logits = vec![0f32; B * T * V];
    for i in 0..B * T * V {
        let byte_4 = iter.next().unwrap();
        let double = f32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        expected_logits[i] = double;
    }

    let byte_4 = iter.next().unwrap();
    let expected_loss = f32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
    println!("Expected loss at step 0: {}", expected_loss);

    let expected_grads_memory = load_params(gpt.param_sizes, iter);
    (
        B,
        T,
        x,
        y,
        expected_logits,
        expected_loss,
        expected_grads_memory,
    )
}

fn init_activations(config: &Config, B: usize, T: usize) -> (usize, [usize; 23]) {
    // Activations
    let mut act_sizes = [0; 23];
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
    num_activations: Option<usize>,
    header: [u32; 256],
    config: Config,
    params: ParameterTensors,
    param_sizes: [usize; 16],
    acts: Option<ActivationTensors>,
    act_sizes: Option<[usize; 23]>,
    // gradients
    grads: ParameterTensors,
    grads_acts: Option<ActivationTensors>,
    // memory for adamw
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
}

impl GPT2 {
    fn forward(&mut self, inputs: &[usize], B: usize, T: usize) -> Vec<f32> {
        let V: usize = self.config.vocab_size;
        let Vp: usize = self.config.padded_vocab_size;
        let L: usize = self.config.num_layers;
        let NH: usize = self.config.num_heads;
        let C: usize = self.config.channels;
        let temperature: f32 = self.config.temperature;

        let acts = self.acts.as_mut().unwrap();
        let params = &mut self.params;

        // forward pass
        encoder_forward(&mut acts.encoded, inputs, &params.wte, &params.wpe, B, T, C);
        for l in 0..L {
            // Input to this layer
            let residual: &[f32] = if l == 0 {
                &acts.encoded // (B, T, C)
            } else {
                &acts.residual3[(l - 1) * B * T * C..l * B * T * C] // Previous layer's output
            };
            layernorm_forward(
                &mut acts.ln1[l * B * T * C..(l + 1) * B * T * C],
                &mut acts.ln1_mean[l * B * T..],
                &mut acts.ln1_rstd[l * B * T..],
                residual,
                &params.ln1w[l * C..],
                &params.ln1b[l * C..],
                B,
                T,
                C,
            );
            matmul_forward(
                &mut acts.qkv[l * B * T * 3 * C..(l + 1) * B * T * 3 * C],
                &acts.ln1[l * B * T * C..],
                &params.qkvw[l * 3 * C * C..],
                Some(&params.qkvb[l * 3 * C..]),
                B,
                T,
                C,
                3 * C,
            );
            attention_forward(
                &mut acts.atty[l * B * T * C..(l + 1) * B * T * C],
                &mut acts.preatt[l * B * NH * T * T..],
                &mut acts.att[l * B * NH * T * T..],
                &acts.qkv[l * B * T * 3 * C..],
                B,
                T,
                C,
                NH,
            );
            matmul_forward(
                &mut acts.attproj[l * B * T * C..(l + 1) * B * T * C],
                &acts.atty[l * B * T * C..],
                &params.attprojw[l * C * C..],
                Some(&params.attprojb[l * C..]),
                B,
                T,
                C,
                C,
            );
            residual_forward(
                &mut acts.residual2[l * B * T * C..(l + 1) * B * T * C],
                residual,
                &acts.attproj[l * B * T * C..],
                B * T * C,
            );
            layernorm_forward(
                &mut acts.ln2[l * B * T * C..(l + 1) * B * T * C],
                &mut acts.ln2_mean[l * B * T..],
                &mut acts.ln2_rstd[l * B * T..],
                &acts.residual2[l * B * T * C..],
                &params.ln2w[l * C..],
                &params.ln2b[l * C..],
                B,
                T,
                C,
            );
            matmul_forward(
                &mut acts.fch[l * B * T * 4 * C..(l + 1) * B * T * 4 * C],
                &acts.ln2[l * B * T * C..],
                &params.fcw[l * 4 * C * C..],
                Some(&params.fcb[l * 4 * C..]),
                B,
                T,
                C,
                4 * C,
            );
            gelu_forward(
                &mut acts.fch_gelu[l * B * T * 4 * C..(l + 1) * B * T * 4 * C],
                &acts.fch[l * B * T * 4 * C..],
                B * T * 4 * C,
            );
            matmul_forward(
                &mut acts.fcproj[l * B * T * C..(l + 1) * B * T * C],
                &acts.fch_gelu[l * B * T * 4 * C..],
                &params.fcprojw[l * 4 * C * C..],
                Some(&params.fcprojb[l * C..]),
                B,
                T,
                4 * C,
                C,
            );
            residual_forward(
                &mut acts.residual3[l * B * T * C..(l + 1) * B * T * C],
                &acts.residual2[l * B * T * C..],
                &acts.fcproj[l * B * T * C..],
                B * T * C,
            );
        }
        layernorm_forward(
            &mut acts.lnf,
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            &acts.residual3[(L - 1) * B * T * C..],
            &params.lnfw,
            &params.lnfb,
            B,
            T,
            C,
        );
        matmul_forward(&mut acts.logits, &acts.lnf, &params.wte, None, B, T, C, Vp);
        softmax_forward(&mut acts.probs, &acts.logits, B, T, V, Vp, temperature);
        // output is (B,T,Vp)
        return acts.probs.clone();
    }

    fn backward(&mut self, inputs: &[usize], targets: &[usize], B: usize, T: usize) {
        let V: usize = self.config.vocab_size;
        let Vp: usize = self.config.padded_vocab_size;
        let L: usize = self.config.num_layers;
        let NH: usize = self.config.num_heads;
        let C: usize = self.config.channels;

        let acts = self.acts.as_mut().unwrap();
        let params = &mut self.params;
        let grads_acts = self.grads_acts.as_mut().unwrap();
        let grads = &mut self.grads;

        // Fill with the mean loss
        let dloss_mean = 1.0 / (B * T) as f32;
        for i in 0..B * T {
            grads_acts.losses[i] = dloss_mean;
        }

        crossentropy_softmax_backward(
            &mut grads_acts.logits,
            &grads_acts.losses,
            targets,
            &acts.probs,
            B,
            T,
            V,
            Vp,
        );
        matmul_backward(
            &mut grads_acts.lnf,
            &mut grads.wte,
            None,
            &grads_acts.logits,
            &acts.lnf,
            &params.wte,
            B,
            T,
            C,
            Vp,
        );
        layernorm_backward(
            &mut grads_acts.residual3[(L - 1) * B * T * C..],
            &mut grads.lnfw,
            &mut grads.lnfb,
            &grads_acts.lnf,
            &acts.residual3[(L - 1) * B * T * C..],
            &acts.lnf_mean,
            &acts.lnf_rstd,
            &params.lnfw,
            B,
            T,
            C,
        );
        let mut residual: &[f32];
        let mut grads_residual: &mut [f32];

        for l in (0..L).rev() {
            residual_backward(
                &mut grads_acts.residual2[l * B * T * C..],
                &mut grads_acts.fcproj[l * B * T * C..],
                &grads_acts.residual3[l * B * T * C..],
                B * T * C,
            );
            matmul_backward(
                &mut grads_acts.fch_gelu[l * B * T * 4 * C..],
                &mut grads.fcprojw[l * C * 4 * C..],
                Some(&mut grads.fcprojb[l * C..]),
                &grads_acts.fcproj[l * B * T * C..],
                &acts.fch_gelu[l * B * T * 4 * C..],
                &params.fcprojw[l * C * 4 * C..],
                B,
                T,
                4 * C,
                C,
            );
            gelu_backward(
                &mut grads_acts.fch[l * B * T * 4 * C..],
                &grads_acts.fch_gelu[l * B * T * 4 * C..],
                &acts.fch[l * B * T * 4 * C..],
                B * T * 4 * C,
            );
            matmul_backward(
                &mut grads_acts.ln2[l * B * T * C..],
                &mut grads.fcw[l * 4 * C * C..],
                Some(&mut grads.fcb[l * 4 * C..]),
                &grads_acts.fch[l * B * T * 4 * C..],
                &acts.ln2[l * B * T * C..],
                &params.fcw[l * 4 * C * C..],
                B,
                T,
                C,
                4 * C,
            );
            layernorm_backward(
                &mut grads_acts.residual2[l * B * T * C..],
                &mut grads.ln2w[l * C..],
                &mut grads.ln2b[l * C..],
                &grads_acts.ln2[l * B * T * C..],
                &acts.residual2[l * B * T * C..],
                &acts.ln2_mean[l * B * T..],
                &acts.ln2_rstd[l * B * T..],
                &params.ln2w[l * C..],
                B,
                T,
                C,
            );
            if l == 0 {
                grads_residual = &mut grads_acts.encoded;
            } else {
                let start = (l - 1) * B * T * C;
                let end = l * B * T * C;
                grads_residual = &mut grads_acts.residual3[start..end];
            }
            residual_backward(
                grads_residual,
                &mut grads_acts.attproj[l * B * T * C..],
                &grads_acts.residual2[l * B * T * C..],
                B * T * C,
            );
            matmul_backward(
                &mut grads_acts.atty[l * B * T * C..],
                &mut grads.attprojw[l * C * C..],
                Some(&mut grads.attprojb[l * C..]),
                &grads_acts.attproj[l * B * T * C..],
                &acts.atty[l * B * T * C..],
                &params.attprojw[l * C * C..],
                B,
                T,
                C,
                C,
            );
            attention_backward(
                &mut grads_acts.qkv[l * B * T * 3 * C..],
                &mut grads_acts.preatt[l * B * NH * T * T..],
                &mut grads_acts.att[l * B * NH * T * T..],
                &grads_acts.atty[l * B * T * C..],
                &acts.qkv[l * B * T * 3 * C..],
                &acts.att[l * B * NH * T * T..],
                B,
                T,
                C,
                NH,
            );

            matmul_backward(
                &mut grads_acts.ln1[l * B * T * C..],
                &mut grads.qkvw[l * 3 * C * C..],
                Some(&mut grads.qkvb[l * 3 * C..]),
                &grads_acts.qkv[l * B * T * 3 * C..],
                &acts.ln1[l * B * T * C..],
                &params.qkvw[l * 3 * C * C..],
                B,
                T,
                C,
                3 * C,
            );
            if l == 0 {
                residual = &acts.encoded;
            } else {
                residual = &acts.residual3[(l - 1) * B * T * C..];
            }
            layernorm_backward(
                grads_residual,
                &mut grads.ln1w[l * C..],
                &mut grads.ln1b[l * C..],
                &grads_acts.ln1[l * B * T * C..],
                residual,
                &acts.ln1_mean[l * B * T..],
                &acts.ln1_rstd[l * B * T..],
                &params.ln1w[l * C..],
                B,
                T,
                C,
            );
        }
        encoder_backward(
            &mut grads.wte,
            &mut grads.wpe,
            &grads_acts.encoded,
            inputs,
            B,
            T,
            C,
        );
    }

    fn loss(&mut self, probs: &[f32], targets: &[usize], B: usize, T: usize) -> f32 {
        // targets is (B,T)
        let Vp: usize = self.config.padded_vocab_size;
        let acts = self.acts.as_mut().unwrap();
        acts.losses = crossentropy_forward(&probs, targets, B, T, Vp);
        let mut mean_loss: f32 = 0.0;
        for loss in &acts.losses {
            mean_loss += loss;
        }
        mean_loss /= (B * T) as f32;
        mean_loss
    }

    fn argmax(&self, probs: Vec<f32>, only_last: bool, B: usize, T: usize) -> Vec<usize> {
        // output is (B, T) if only_last=false
        // (B) if only_last=true
        // get the token with the highest probability for each dimension (temperature=1)
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
        tokenizer: &tokenizer::Tokenizer,
        N: usize,
        B: usize,
        T: usize,
    ) -> Vec<usize> {
        // inputs (B, T)
        // outputs (B, T + N)
        let mut modified_inputs = inputs.clone();
        let mut outputs = inputs.clone();
        for i in 0..N {
            // inputs is (1,T)
            // probs is (B, T, Vp)
            let probs: Vec<f32> = self.forward(&modified_inputs, B, T);

            // (B) due to only_last=true
            let tokens = self.argmax(probs, true, B, T);

            // consider only the first batch element
            for b in 0..1 {
                // Insert at the end of the batched input
                outputs.insert((b + 1) * (T + i), tokens[b]);
                // Insert at the end of the sentence
                modified_inputs.insert((b + 1) * T, tokens[b]);
                // Remove the first token
                modified_inputs.remove(b * T);
            }
        }
        outputs
    }

    fn generate(
        &mut self,
        inputs: Vec<usize>,
        N: usize,
        B: usize,
        T: usize,
        tokenizer: &tokenizer::Tokenizer,
    ) {
        //TODO: ignore B actually
        let input_text = tokenizer.batch_decode(&inputs[0..T]);
        println!("Input text: {}", input_text);
        self.sample(inputs, tokenizer, N, B, T);
    }

    fn zero_grad(&mut self) {
        self.grads_acts = Some(init_acts(self.act_sizes.unwrap()));
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
                // Compute bias-corrected estimates using powf to avoid potential integer overflow
                let m_hat: f32 = m / (1.0 - beta1.powf(t as f32));
                // Compute bias-corrected second raw moment estimate
                let v_hat: f32 = v / (1.0 - beta2.powf(t as f32));

                // Persist optimizer state
                m_memory[i] = m;
                v_memory[i] = v;

                // AdamW parameter update (decoupled weight decay)
                *param *= 1.0 - learning_rate * weight_decay;
                *param -= learning_rate * (m_hat / (v_hat.sqrt() + eps));

                i += 1;
            }
        }
    }
}

fn train(
    model_path: &str,
    tokenizer_path: &str,
    train_data_path: &str,
    val_data_path: &str,
    B: usize,
    T: usize,
    should_suffle: bool,
) {
    println!("Running train...");
    let mut train_dataloader = dataloader::create_dataloader(train_data_path, B, T, should_suffle);
    let mut val_dataloader = dataloader::create_dataloader(val_data_path, B, T, false);

    let tokenizer = tokenizer::load_tokenizer(&tokenizer_path);

    println!("train dataset num_batches: {}", train_dataloader.nbatch);
    println!("val dataset num_batches: {}", val_dataloader.nbatch);
    let val_num_batches = 5;

    let temperature: f32 = 1.0;
    let mut gpt = load_model(&model_path, temperature);
    load_acts(&mut gpt, B, T);

    let genT: usize = 64;
    let Vp = gpt.config.padded_vocab_size;
    let V = gpt.config.vocab_size;

    for step in 0..=40 {
        // validate every 10 step
        if step % 10 == 0 {
            let mut val_loss = 0f32;
            val_dataloader.reset();
            for _ in 0..val_num_batches {
                val_dataloader.next_batch();
                let probs = gpt.forward(&val_dataloader.inputs, B, T);
                let mean_loss: f32 = gpt.loss(&probs, &val_dataloader.targets, B, T);
                val_loss += mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("Validation loss {} at step {}", val_loss, step);
        }

        // generate data every 20 step
        if step > 0 && step % 20 == 0 {
            println!("Generating {} tokens...", genT);
            let mut gen_tokens = vec![tokenizer.eot_token; B * T];
            for t in 1..genT {
                let probs = gpt.forward(&gen_tokens, B, T);
                // probs (B,T,Vp)
                // sample w.r.t the probabilities
                let token_id = utils::sample_mult(&probs[(t - 1) * Vp..(t - 1) * Vp + V], V);
                gen_tokens[t] = token_id;
            }

            let gen_text = tokenizer.batch_decode(&gen_tokens[1..genT]);
            println!("Generated text: '{}'", gen_text);
        }

        // get data batch
        train_dataloader.next_batch();
        let now = SystemTime::now();
        // forward pass
        let probs: Vec<f32> = gpt.forward(&train_dataloader.inputs, B, T);
        let mean_loss: f32 = gpt.loss(&probs, &train_dataloader.targets, B, T);
        // zero grad
        gpt.zero_grad();
        // backward pass
        gpt.backward(&train_dataloader.inputs, &train_dataloader.targets, B, T);
        // update
        gpt.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);

        let time_elapsed_ms = now.elapsed().unwrap().as_millis();
        println!(
            "step {}: loss {} (took {} ms)",
            step, mean_loss, time_elapsed_ms
        );
    }
}

fn test(model_path: &str, debug_path: &str) {
    println!("Running test...");
    let temperature: f32 = 1.0;
    let mut gpt = load_model(&model_path, temperature);
    let (B, T, x, y, expected_logits, expected_loss, expected_grads) =
        load_debug_state(debug_path, &gpt);
    load_acts(&mut gpt, B, T);

    let expected_losses = [
        5.270007133483887,
        4.059706687927246, // 4.0593915
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615,
    ];

    let V = gpt.config.vocab_size;
    let L = gpt.config.num_layers;
    let C = gpt.config.channels;
    let maxT = gpt.config.max_seq_len;
    let Vp = gpt.config.padded_vocab_size;
    let eps = 1e-2;

    for step in 0..10 {
        let now = SystemTime::now();

        let probs: Vec<f32> = gpt.forward(&x, B, T);
        let mean_loss: f32 = gpt.loss(&probs, &y, B, T);
        gpt.zero_grad();
        gpt.backward(&x, &y, B, T);
        gpt.update(1e-4, 0.9, 0.999, 1e-8, 0.01, step + 1);

        let time_elapsed_ms = now.elapsed().unwrap().as_millis();
        println!(
            "step {}: loss {} (took {} ms)",
            step, mean_loss, time_elapsed_ms
        );

        if step == 0 {
            // Check loss
            if (mean_loss - expected_loss).abs() >= eps {
                panic!(
                    "Losses aren't the same (expected={}, got={})",
                    expected_loss, mean_loss
                );
            } else {
                println!("OK (LOSS)");
            }
            // Check logits
            let calculated_logits = &gpt.acts.as_ref().unwrap().logits;
            let mut max_diff = 0f32;
            for bt in 0..B * T {
                for v in 0..V {
                    let ix = bt * Vp + v;
                    let diff = (calculated_logits[ix] - expected_logits[bt * V + v]).abs();
                    max_diff = max_diff.max(diff);
                    if diff >= eps {
                        println!(
                            "expected={}, got={}",
                            expected_logits[bt * V + v],
                            calculated_logits[ix]
                        );
                        panic!("Logits aren't the same at index {},{}", bt, v);
                    }
                }
            }
            println!("OK (LOGITS), max_diff={}", max_diff);

            // Check gradients
            let grads = &gpt.grads;
            utils::check_tensor(&grads.wte, &expected_grads.wte, V * C, "dwte");
            utils::check_tensor(&grads.wpe, &expected_grads.wpe, maxT * C, "dwpe");
            utils::check_tensor(&grads.ln1w, &expected_grads.ln1w, L * C, "dln1w");
            utils::check_tensor(&grads.ln1b, &expected_grads.ln1b, L * C, "dln1b");
            utils::check_tensor(&grads.qkvw, &expected_grads.qkvw, L * 3 * C * C, "dqkvw");
            utils::check_tensor(&grads.qkvb, &expected_grads.qkvb, L * 3 * C, "dqkvb");
            utils::check_tensor(
                &grads.attprojw,
                &expected_grads.attprojw,
                L * C * C,
                "dattprojw",
            );
            utils::check_tensor(
                &grads.attprojb,
                &expected_grads.attprojb,
                L * C,
                "dattprojb",
            );
            utils::check_tensor(&grads.ln2w, &expected_grads.ln2w, L * C, "dln2w");
            utils::check_tensor(&grads.ln2b, &expected_grads.ln2b, L * C, "dln2b");
            utils::check_tensor(&grads.fcw, &expected_grads.fcw, L * 4 * C * C, "dfcw");
            utils::check_tensor(&grads.fcb, &expected_grads.fcb, L * 4 * C, "dfcb");
            utils::check_tensor(
                &grads.fcprojw,
                &expected_grads.fcprojw,
                L * C * 4 * C,
                "dfcprojw",
            );
            utils::check_tensor(&grads.fcprojb, &expected_grads.fcprojb, L * C, "dfcprojb");
            utils::check_tensor(&grads.lnfw, &expected_grads.lnfw, C, "dlnfw");
            utils::check_tensor(&grads.lnfb, &expected_grads.lnfb, C, "dlnfb");
        }
        // Check loss
        let exp_loss = expected_losses[step];
        assert!(
            (mean_loss - exp_loss).abs() < eps,
            "Losses aren't the same (expected={}, got={})",
            exp_loss,
            mean_loss
        );
    }
}

fn generate(
    model_path: &str,
    tokenizer_path: &str,
    input: Vec<usize>,
    T: usize,
    temperature: f32,
    N: usize,
) {
    // Sample N tokens
    let B: usize = 1;

    let tokenizer = tokenizer::load_tokenizer(&tokenizer_path);

    let mut gpt = load_model(&model_path, temperature);
    load_acts(&mut gpt, 1, T);

    gpt.generate(input, N, B, T, &tokenizer);
}

fn main() {
    let B: usize = 4;
    let T: usize = 64;
    let should_suffle: bool = true;

    let train_data_path = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    let val_data_path = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    let model_path = "gpt2_124M.bin";
    let debug_path = "gpt2_124M_debug_state.bin";
    let tokenizer_path = "gpt2_tokenizer.bin";

    train(
        model_path,
        tokenizer_path,
        train_data_path,
        val_data_path,
        B,
        T,
        should_suffle,
    );

    // test(model_path, debug_path);
}
