pub struct DataLoader {
    dataset: Vec<usize>,
    curr_idx: usize,
    pub len: usize,
    B: usize, // batch size
    T: usize, // sentence length
    pub inputs: Vec<usize>,
    pub targets: Vec<usize>,
}

impl DataLoader {
    pub fn new(dataset: Vec<usize>, B: usize, T: usize) -> DataLoader {
        let len = dataset.len();
        DataLoader {
            dataset,
            curr_idx: 0,
            len,
            B,
            T,
            inputs: Vec::new(),
            targets: Vec::new(),
        }
        //todo: implement random batch
    }

    pub fn next_batch(&mut self) {
        // We need one more token for the last target
        if self.curr_idx + (self.T * self.B) + 1 > self.len {
            self.reset();
        }
        self.load_batch();

        // Increase self.curr_idx
        self.curr_idx += self.T * self.B // The last token can be use as input
    }

    pub fn load_batch(&mut self) {
        self.inputs = Vec::new();
        self.targets = Vec::new();

        let end: usize = self.curr_idx + self.T * self.B;

        for i in self.curr_idx..end {
            self.inputs.push(self.dataset[i]);
            self.targets.push(self.dataset[i + 1]);
        }
    }

    pub fn reset(&mut self) {
        self.curr_idx = 0;
    }
}

pub fn create_dataloader(data_path: &str, B: usize, T: usize) -> DataLoader {
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

    DataLoader::new(dataset, B, T)
}
