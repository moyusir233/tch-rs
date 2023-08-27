lto_rust_flags:="-Clink-arg=-fuse-ld=lld-16 -Clinker=clang-16 -Clinker-plugin-lto -Clink-arg=-Wl,--allow-shlib-undefined --cfg enable_lto"

# 测试lto优化,RUSTFLAGS中传递给编译器的各个参数表示使用lld-16链接器、使用clang-16链接、开启lto plugin以进行跨语言的lto优化、以及在build script中使用的用于开启cmake侧的lto优化的自定义cfg enable_lto，详见:https://doc.rust-lang.org/rustc/linker-plugin-lto.html
[no-cd]
test-lto *args:
    RUSTFLAGS="{{lto_rust_flags}}" cargo test {{args}}

[no-cd]
clean:
    cargo clean