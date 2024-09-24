use core::f32;

// T = sequence_length
// B = batch_size
// C = channels
// V = vocab_size

fn matmul_forward(input: &[f32], weight: &[f32], bias: &[f32], B: usize, T: usize, C: usize, O: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(B*T*O);
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;
            for o in 0..O {
                let mut val: f32 = bias[o];
                for i in 0..C {
                    val += input[bt * C + i] * weight[o * C + i];
                }
                output.push(val);
            }
        }
    }
    output
}

fn gelu_forward(input: &[f32], N: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        let x = input[i];
        let cube = 0.44715 * x * x * x;
        output.push(0.5 * x * (1.0 + f32::tanh(f32::sqrt(2.0/f32::consts::PI)*(x + cube))))
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

fn attention_forward(preatt: &[f32], att: &[f32], inp: &[f32], B: usize, T: usize, C: usize, NH: usize) -> Vec<f32> {
    // input (B,T,3C) holds (Q,K,V) values
    // preatt, att (B, NH, T, T)
    // NH = number of heads
    // T = sequence length
    // output is (B,T,C)
    let mut output: Vec<f32> = Vec::with_capacity(B*T*C);
    output
}

fn encoder_forward(input: &[u32], wte: &[f32], wpe: &[f32], B: usize, T: usize, C: usize) -> Vec<f32> {
    // input (B,T,C)
    // wte (V,C): weight token embedding
    // V is vocabulary size
    // wpe (maxT, C): weight positional embedding
    let mut output = Vec::with_capacity(B*T*C);
    for b in 0..B {
        for t in 0..T {
            // Get token_id
            let token_id: usize = input[b * T + t].try_into().unwrap();
            for c in 0..C {
                output.push(wte[token_id * C + c] + wpe[t * C + c]);
            }
        }
    }
    output
}

fn softmax_forward(logits: &[f32], B: usize, T: usize, V: usize, Vp: usize) -> Vec<f32> {
    // input (B,T,V)
    // output (B,T,V)
    // Vp is the padded vocab size
    // V is the real vocab size
    let mut output: Vec<f32> = Vec::with_capacity(B*T*V);
    for b in 0..B {
        for t in 0..T {
            let bt = b * T + t;
            let mut sum: f32 = 0.0;
            for v in 0..V {
                sum += f32::exp(logits[bt + v]);
            }
            for v in 0..V {
                output.push(f32::exp(logits[bt + v]) / sum);
            }
        }
    }
    output
}

fn crossentropy_forward(probs: &[f32], targets: &[f32], B: usize, T: usize, Vp: usize) -> Vec<f32> {
    let output: Vec<f32> = Vec::with_capacity(B*T*Vp);
    for b in 0..B {
        
    }
    output
}

fn main() {
    let input: Vec<f32> = vec![-1.0,2.0,3.0,4.0,5.0,6.0,7.0,9.0];
    let weight: Vec<f32> = vec![1.0,0.0];
    let bias: Vec<f32> = vec![0.5];

    let B: usize = 2;
    let T: usize = 2;
    let C: usize = 2;
    let OC: usize = 1;

    let output = matmul_forward(&input, &weight, &bias, B, T, C, OC);
    println!("Matmul {:?}", output);

    let output = softmax_forward(&input, B, T, C, OC);
    println!("Softmax {:?}", output);

    let output = gelu_forward(&input, B*T*C);
    println!("GELU {:?}", output);
}
