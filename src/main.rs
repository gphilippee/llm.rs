use core::f32;
use std::io::{self, Write};

mod tokenizer;
mod dataloader;

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

fn fast_matmul_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    O: usize,
) {
    //todo: implement
}

fn gelu_forward(input: &[f32], N: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        let x = input[i];
        let cube = 0.44715 * x * x * x;
        output.push(0.5 * x * (1.0 + f32::tanh(f32::sqrt(2.0 / f32::consts::PI) * (x + cube))))
    }
    output
}

fn residual_forward(input1: &[f32], input2: &[f32], N: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        output.push(input1[i] + input2[i])
    }
    output
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
    // output is (B,T,C)
    let mut output: Vec<f32> = Vec::with_capacity(B * T * C);
    let C3 = 3 * C;
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_idx = b * T * C3 + t * C3 + h * hs;

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
                    // TODO: fix
                    preatt[t2] = val;
                }

                // pass2: calcul the exp and keep track of sum
                let mut expsum: f32 = 0.0;
                for t2 in 0..t {
                    let expv = (preatt[t2] - maxval).exp();
                    expsum += expv;
                    // TODO: fix
                    att[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // pass3: normalize to get the softmax
                for t2 in 0..T {
                    if t2 < t {
                        att[t2] *= expsum_inv;
                    } else {
                        att[t2] = 0.0;
                    }
                }
                // pass4: accumulate wieghted values into the output of attention
                for _ in 0..hs {
                    output.push(0.0);
                }
                for t2 in 0..t {
                    let value_idx = b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    for i in 0..hs {
                        output[i] = att[t2] * inp[value_idx + i];
                    }
                }
            }
        }
    }
    output
}

fn encoder_forward(
    input: &[usize],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) -> Vec<f32> {
    // input (B,T,C)
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
    let mut probs: Vec<f32> = Vec::with_capacity(B * T * Vp);
    for b in 0..B {
        for t in 0..T {
            let btv = b * T * Vp + t * Vp;
            let mut sum: f32 = 0.0;
            for v in 0..V {
                sum += f32::exp(logits[btv + v] / temperature);
            }
            for v in 0..V {
                probs.push(f32::exp(logits[btv + v] / temperature) / sum);
            }
            // Pad probabilities
            for _ in V..Vp {
                probs.push(0f32);
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

pub struct GPT2<'a> {
    num_parameters: usize,
    num_activations: usize,
    header: [u32; 256],
    params: ParameterTensors<'a>,
    params_memory: Vec<f32>,
    param_sizes: [usize; 16],
    acts: ActivationTensors<'a>,
    acts_memory: Vec<f32>,
    act_sizes: [usize; 23],
    config: Config,
    // gradients
    grads: Option<ParameterTensors<'a>>,
    grads_memory: Option<Vec<f32>>,
    grads_acts: Option<ActivationTensors<'a>>,
    grads_acts_memory: Option<Vec<f32>>,
    // memory for adamw
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
}

impl GPT2<'_> {
    fn forward(&mut self, inputs: &[usize], temperature: f32) -> Vec<f32>{
        let B: usize = self.config.B;
        let T: usize = self.config.T;
        let V: usize = self.config.vocab_size;
        let Vp: usize = self.config.padded_vocab_size;
        let L: usize = self.config.num_layers;
        let NH: usize = self.config.num_heads;
        let C: usize = self.config.channels;

        let mut acts = self.acts.as_mut().unwrap();
        let params = self.params.as_ref().unwrap();

        // forward pass
        let mut residual: &[f32];
        acts.encoded = encoder_forward(inputs, &params.wte, &params.wpe, B, T, C);
        for l in 0..L {
            // println!("Layer n°{}", l);
            if l == 0 {
                residual = &acts.encoded;
            } else {
                residual = &acts.residual3;
            }
            acts.ln1 = layernorm_forward(
                &mut acts.ln1_mean,
                &mut acts.ln1_rstd,
                residual,
                &params.ln1w,
                &params.ln1b,
                B,
                T,
                C,
            );
            acts.qkv = matmul_forward(&acts.ln1, &params.qkvw, Some(&params.qkvb), B, T, C, 3 * C);
            acts.atty = attention_forward(&mut acts.preatt, &mut acts.att, &acts.qkv, B, T, C, NH);
            acts.attproj = matmul_forward(
                &acts.atty,
                &params.attprojw,
                Some(&params.attprojb),
                B,
                T,
                C,
                C,
            );
            acts.residual2 = residual_forward(&residual, &acts.attproj, B * T * C);
            acts.ln2 = layernorm_forward(
                &mut acts.ln2_mean,
                &mut acts.ln2_rstd,
                &acts.residual2,
                &params.ln2w,
                &params.ln2b,
                B,
                T,
                C,
            );
            acts.fch = matmul_forward(&acts.ln2, &params.fcw, Some(&params.fcb), B, T, C, 4 * C);
            acts.fch_gelu = gelu_forward(&acts.fch, B * T * 4 * C);
            acts.fcproj = matmul_forward(
                &acts.fch_gelu,
                &params.fcprojw,
                Some(&params.fcprojb),
                B,
                T,
                4 * C,
                C,
            );
            acts.residual3 = residual_forward(&acts.residual2, &acts.fcproj, B * T * C);
        }
        // println!("Layers applied");
        acts.lnf = layernorm_forward(
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            &acts.residual3,
            &params.lnfw,
            &params.lnfb,
            B,
            T,
            C,
        );
        acts.logits = matmul_forward(&acts.lnf, &params.wte, None, B, T, C, Vp);
        acts.probs = softmax_forward(&acts.logits, B, T, V, Vp, temperature);
        // output is (B,T,V)
        return acts.probs.clone();
    }

    fn loss(&mut self, probs: &[f32], targets: &[usize]) -> f32 {
        // targets is (B,T)
        let B: usize = self.config.B;
        let Vp: usize = self.config.padded_vocab_size;
        let V: usize = self.config.vocab_size;
        let T: usize = self.config.T;
        let mut acts = self.acts.as_mut().unwrap();
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

    fn backward(&self) {
        //todo: implement backpropagation
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
            token_ids = Vec::with_capacity(B);
            for b in 0..B {
                let start_idx = b * T * Vp + (T - 1) * Vp;
                let end_idx = b * T * Vp + T * Vp;
                let token_id = probs[start_idx..end_idx]
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.partial_cmp(y).expect("Partial comparison paniced"))
                    .map(|(index, _)| index)
                    .expect("No index was found");
                token_ids.push(token_id);
            }
        } else {
            token_ids = Vec::with_capacity(B * T);
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
                    token_ids.push(token_id);
                }
            }
        }
        token_ids
    }

    fn sample(&mut self, inputs: Vec<usize>, N: usize, temperature: f32, tokenizer: &tokenizer::Tokenizer) -> Vec<usize> {
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

            // probs is (B, T, V)
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
                    let token = tokenizer.decode(tokens[b]);
                    print!("{}", token);
                    io::stdout().flush().unwrap();  // Force flush
                }
            }
        }
        outputs
    }

    fn generate(&mut self, inputs: Vec<usize>, N: usize, temperature: f32, tokenizer: &tokenizer::Tokenizer) {
        let B = self.config.B;
        let T = self.config.T;
        //TODO: ignore B actually
        let input_text = tokenizer.batch_decode(&inputs[0..T]);
        println!("Input text: {}", input_text);
        self.sample(inputs, N, temperature, tokenizer);
    }

    fn zero_grad(&mut self) {
        self.grads_acts_memory = Some(vec![0f32; self.num_activations]);
        self.grads_memory = Some(vec![0f32; self.num_parameters])
    }

    fn update(&mut self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: usize) {
    //     // implement adamw algorithm
    //     for i in 0..self.num_parameters.expect("Missing total parameters!") {
    //         let param: f32 = self.params_memory.unwrap()[i];
    //         // Get gradients w.r.t. stochastic objective at timestep t
    //         let grad: f32 = self.grads_memory.unwrap()[i];
    //         // Update biased first moment estimate
    //         let m: f32 = beta1 * self.m_memory.unwrap()[i] + (1.0 - beta1) * grad;
    //         // Update biased second raw moment estimate
    //         let v: f32 = beta2 * self.v_memory.unwrap()[i] + (1.0 - beta2) * grad.powi(2);
    //         // Compute bias-corrected first moment estimate
    //         let m_hat: f32 = m / (1.0 - beta1);
    //         // Compute bias-corrected second raw moment estimate
    //         let v_hat: f32 = v / (1.0 - beta2);
    //         // Update parameters
    //         self.m_memory.unwrap()[i] = m;
    //         self.v_memory.unwrap()[i] = v;
    //         //todo: add sqrt on v
    //         self.params_memory.unwrap()[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
    //     }
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

    let mut gpt = GPT2::new(model_header, config);

    let mut param_sizes = [0; 16];

    let Vp = gpt.config.padded_vocab_size;
    let C = gpt.config.channels;
    let maxT = gpt.config.max_seq_len;
    let L = gpt.config.num_layers;

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
    gpt.param_sizes = Some(param_sizes);

    // println!("{:?}", param_sizes);

    // Total parameters
    let mut num_parameters: usize = 0;
    for n_params in param_sizes {
        num_parameters += n_params;
    }
    println!("Total parameters: {}M", num_parameters as u32 / 10u32.pow(6));
    gpt.num_parameters = Some(num_parameters);

    // Init 1st and 2nd moments
    gpt.params_memory = Some(vec![0f32; num_parameters]);
    gpt.m_memory = Some(vec![0f32; num_parameters]);
    gpt.v_memory = Some(vec![0f32; num_parameters]);

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

    let mut gpt2 = GPT2 {
        num_parameters,
        num_activations,
        header: model_header,
        config,
        params,
        params_memory,
        param_sizes,
        acts,
        acts_memory,
        act_sizes,
        grads_acts: None,
        grads_acts_memory: None,
        grads: None,
        grads_memory: None,
        m_memory: vec![0f32; num_parameters],
        v_memory: vec![0f32; num_parameters],
    };
    
    gpt2
}

fn init_activations(gpt: &mut GPT2) -> ActivationTensors {
    // Activations
    let mut act_sizes = [0; 23];
    let B: usize = gpt.config.B; // batch size
    let T: usize = gpt.config.T; // seq_len
    let C: usize = gpt.config.channels;
    let NH: usize = gpt.config.num_heads;
    let L: usize = gpt.config.num_layers;
    let Vp: usize = gpt.config.padded_vocab_size;

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
    gpt.num_activations = Some(num_activations);
    println!("Total activations: {}M", num_activations / 10usize.pow(6));

    ActivationTensors {
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
    }
}

fn main() {
    let B: usize = 4;
    let T: usize = 64;
    let temperature: f32 = 1.0;
    let tokens_to_sample: usize = 10;

    let train_data_path = "/Users/gphilippe/dev/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    let val_data_path = "/Users/gphilippe/dev/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    let model_path = "/Users/gphilippe/dev/llm.c/gpt2_124M.bin";
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
    // let batch_inputs = inputs[0..B * T].to_vec();
    // gpt.generate(batch_inputs, tokens_to_sample, temperature, &tokenizer);

    //todo: code for batching inputs and targets
    // let n_inputs = inputs.len() / T;
    // println!("Number of inputs {}", n_inputs);
    // for (i, n) in (0..n_inputs).step_by(B).enumerate() {
    //     let start_idx = n * T;
    //     let end_idx = ((n + B) * T).min(inputs.len());
    //     println!("Batch {} - Start {} - End {}", i, start_idx, end_idx);
    //     // Wrong way to batch
    //     let batch_inputs = &inputs[start_idx..end_idx];
    //     let batch_targets = &targets[start_idx..end_idx];
    //     gpt.loss(batch_inputs, batch_targets, temperature);
    //     break;
    // }

    // todo: implement the train loop
    // how inputs are batched ?
    // implement backward methods
    // compare loss at first iteration

    for step in 0..40 {
        // get data batch
        train_dataloader.next_batch();
        // forward pass
        println!("Inputs len: {}", &train_dataloader.inputs.len());
        let probs: Vec<f32> = gpt.forward(&train_dataloader.inputs, temperature);
        let loss: f32 = gpt.loss(&probs, &train_dataloader.targets);
        println!("Loss {} at step {}", loss, step);
        // zero grad
        gpt.zero_grad();
        // backward pass
        gpt.backward();
        // update
        gpt.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
    }
}
