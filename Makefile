default:
	cargo build --release
install:
	cp target/release/db-respawn-extractor ~/bin/
