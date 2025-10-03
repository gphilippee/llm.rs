# llm.rs

Implementation of GPT-2 using Rust. 

Migration of [llm.c](https://github.com/karpathy/llm.c) from the great Andrej Karpathy.

## Unsafe Code Notice

This project uses unsafe code exclusively for matrix multiplication operations (forward and backward passes). The rest of the codebase is implemented with safe practices. Unsafe blocks are kept minimal and well-contained to ensure performance without compromising overall safety.

## Quick start

1. Download data and models
```bash
make setup
```

2. Build the project
```bash
make build
```

3. Run the training script
```bash
make run
```

## Performance

Every training step takes around 1200ms. 

This performance could be improved by using more `unsafe` code.

## Contributing

Contributions are welcome! Whether it’s fixing bugs, adding new features, improving documentation, or suggesting ideas—your input helps this project grow.