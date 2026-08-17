fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "grpc")]
    {
        // Vendored protoc so a checkout needs no protobuf toolchain installed
        // to build the optional feature.
        // SAFETY: build scripts are single-threaded; nothing else reads the env.
        unsafe {
            std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
        }
        println!("cargo:rerun-if-changed=proto/memvault.proto");
        tonic_prost_build::compile_protos("proto/memvault.proto")?;
    }
    Ok(())
}
