#!/bin/bash
cargo test process_rank0 -- --nocapture &
cargo test process_rank1 -- --nocapture &
wait