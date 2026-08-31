use o3gen::Generator;

fn main() {
    // Generate types from petstore OpenAPI
    Generator::builder("petstore.json")
        .rename("Category", "PetCategory")
        .api_name("PetApi")
        .write_to_out_dir("types.rs")
        .expect("Failed to generate types.rs");

    Generator::builder("fixtures/composite.json")
        .deny_unknown_fields(true)
        .write_to_out_dir("composite.rs")
        .expect("Failed to generate composite.rs");

    Generator::builder("fixtures/newtypes.json")
        .write_to_out_dir("newtypes.rs")
        .expect("Failed to generate newtypes.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=petstore.json");
    println!("cargo:rerun-if-changed=fixtures/composite.json");
    println!("cargo:rerun-if-changed=fixtures/newtypes.json");
}
