mod f16_ty {
    type TestEG = half::f16;
    include!("suite.rs");
}

mod f32_ty {
    type TestEG = f32;
    include!("suite.rs");
}
