use crate::utils;

pub struct DataLoader {
    dataset: Vec<usize>,
    current_batch: usize,
    pub ntok: usize,
    pub nbatch: usize,
    B: usize, // batch size
    T: usize, // sentence length
    pub inputs: Vec<usize>,
    pub targets: Vec<usize>,
    shuffle: bool,
    permutation: Vec<usize>,
}

impl DataLoader {
    pub fn new(dataset: Vec<usize>, B: usize, T: usize, shuffle: bool) -> DataLoader {
        let ntok = dataset.len();
        assert!(ntok > 0, "Dataset is empty!");

        let nbatch = ntok / (B * T);

        let mut permutation = utils::init_random_permutation(nbatch);
        if shuffle {
            println!("Dataloader is shuffled and will be shuffled every times `reset` is call");
            utils::random_permutation(&mut permutation, nbatch);
        }

        DataLoader {
            dataset: dataset,
            current_batch: 0,
            ntok,
            nbatch,
            B,
            T,
            inputs: Vec::new(),
            targets: Vec::new(),
            shuffle,
            permutation,
        }
    }

    fn prepare_permutation(&self) -> Vec<usize> {
        let mut permutation = utils::init_random_permutation(self.nbatch);
        utils::random_permutation(&mut permutation, self.nbatch);
        permutation
    }

    pub fn next_batch(&mut self) {
        if self.current_batch >= self.nbatch {
            self.reset();
        }
        self.load_batch();

        // Increase current batch
        self.current_batch += 1
    }

    pub fn load_batch(&mut self) {
        self.inputs = Vec::new();
        self.targets = Vec::new();

        let permuted_batch = self.permutation[self.current_batch];
        let idx_offset = permuted_batch * self.B * self.T;

        for i in 0..(self.B * self.T) {
            self.inputs.push(self.dataset[idx_offset + i]);
            self.targets.push(self.dataset[idx_offset + i + 1]);
        }
    }

    pub fn reset(&mut self) {
        self.current_batch = 0;

        if self.shuffle {
            self.prepare_permutation();
        }
    }
}

pub fn create_dataloader(data_path: &str, B: usize, T: usize, shuffle: bool) -> DataLoader {
    // Load inputs
    let bytes = std::fs::read(data_path).unwrap();

    let mut iter = bytes.chunks_exact(2);
    let mut dataset: Vec<usize> = Vec::new();
    // Data header (skip it)
    for _ in 0..512 {
        iter.next().unwrap();
    }

    for bytes in iter {
        let token_id = u16::from_be_bytes([bytes[1], bytes[0]]);
        dataset.push(token_id.into())
    }

    DataLoader::new(dataset, B, T, shuffle)
}
