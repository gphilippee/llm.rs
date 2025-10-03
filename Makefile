setup:
	./dev/download_starter_pack.sh

build:
	cargo build --release

run:
	./target/release/llm

clean:
	cargo clean