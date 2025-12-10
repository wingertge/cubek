use cubecl::prelude::*;
use cubecl::std::FastDivmod;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod,
}
