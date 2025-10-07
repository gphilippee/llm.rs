pub struct Tokenizer {
    vocab_size: usize,
    pub eot_token: usize,
    token_table: Option<Vec<String>>,
}

impl Tokenizer {
    pub fn check(&self, dataset: &Vec<usize>) {
        // validate inputs: check if all tokens are between [0, V)
        let vocab_size = self.vocab_size;
        for token_id in dataset {
            assert!(
                *token_id < vocab_size,
                "Token {token_id} is superior to {vocab_size}"
            );
        }
    }

    pub fn decode(&self, token_id: usize) -> &String {
        if token_id < self.vocab_size {
            // Access the token table safely
            if let Some(token_table) = &self.token_table {
                if let Some(token) = token_table.get(token_id) {
                    return token;
                }
            }
            panic!("Invalid token_id: token {token_id} not found.");
        } else {
            panic!("`token_id` must be smaller than vocab_size");
        }
    }

    pub fn batch_decode(&self, tokens: &[usize]) -> String {
        let mut output = String::new();
        for t in tokens {
            let substr = self.decode(*t);
            output += substr;
        }
        output
    }
}

pub fn load_tokenizer(file: &str) -> Tokenizer {
    // Load tokenizer
    let bytes = std::fs::read(file).unwrap();

    let mut iter = bytes.chunks_exact(4);
    let mut tokenizer_header: [u32; 256] = [0; 256];
    for i in 0..256 {
        let byte_4 = iter.next().unwrap();
        let double = u32::from_be_bytes([byte_4[3], byte_4[2], byte_4[1], byte_4[0]]);
        tokenizer_header[i] = double;
    }

    let mut tok = Tokenizer {
        vocab_size: tokenizer_header[2] as usize,
        eot_token: tokenizer_header[3] as usize,
        token_table: None,
    };

    let mut token_table: Vec<String> = Vec::new();
    let mut iter = bytes[1024..].chunks(1);
    for _ in 0..tok.vocab_size {
        let byte_4 = iter.next().unwrap();
        let length: u8 = byte_4[0];

        let mut token = String::new();
        for _ in 0..length {
            let byte = iter.next().unwrap()[0];
            let c: char = char::from(byte);
            token.push(c);
        }
        token_table.push(token);
    }
    assert!(
        token_table.len() == tok.vocab_size,
        "length are not equals token_table={}, vocab_size= {}",
        token_table.len(),
        tok.vocab_size
    );
    tok.token_table = Some(token_table);

    tok
}
