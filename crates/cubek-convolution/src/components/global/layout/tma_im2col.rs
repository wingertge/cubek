use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::layout::{Coords3d, CoordsDyn, Layout, LayoutExpand},
    },
};

use crate::components::{ConvolutionParams, global::layout::NhwcCoords};

/// Im2col layout, producing both the position and offset
#[derive(CubeType, CubeLaunch)]
pub struct TmaIm2colLayout {
    shape_out: Sequence<FastDivmod>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    params: ConvolutionParams,
}

#[cube]
impl TmaIm2colLayout {
    pub fn new(
        shape_out: Sequence<FastDivmod>,
        padded_channels: FastDivmod,
        #[comptime] params: ConvolutionParams,
    ) -> Self {
        TmaIm2colLayout {
            shape_out,
            padded_channels,
            params,
        }
    }
}

#[cube]
impl Layout for TmaIm2colLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = (NhwcCoords, CoordsDyn);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, m, k) = pos;
        let params = comptime![self.params];

        let (n_offs, spatial_offsets) = div_mod_seq(m, &self.shape_out);
        let spatial_dims = spatial_offsets.len();

        let mut in_offs = Sequence::<i32>::new();

        comptime![println!("params: {params:?}")];

        #[unroll]
        for dim in 0..spatial_dims {
            let offs = spatial_offsets.index(dim) * comptime![params.stride[dim as usize]];
            let offs = offs as i32 - comptime![params.padding[dim as usize]];
            in_offs.push(offs);
        }

        let (mut k_idx, channel_start) = self.padded_channels.div_mod(k);

        let pos = NhwcCoords {
            batch: n_offs,
            spatial: in_offs,
            channel: channel_start,
        };

        let mut k_offs = Sequence::new();
        let k_rank = params.dimensionality.num_dims();

        #[unroll]
        for i in 0..k_rank {
            let dim = comptime![(k_rank - i - 1) as usize];
            let k_size = comptime!(params.kernel_size[dim]);
            k_offs.push((k_idx % k_size) * comptime!(params.dilation[dim]));
            k_idx /= k_size;
        }

        (pos, k_offs.rev())
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (u32::MAX, u32::MAX, u32::MAX).runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

/// Decompose a linear index into local positions along each dimension in `shape`. Also returns the
/// left over remainder.
#[cube]
pub(crate) fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}
