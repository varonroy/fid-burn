use std::collections::HashSet;

use burn::prelude::*;
use inception_v3_burn::model::InceptionV3;

pub trait FidForward<B: Backend> {
    fn fid_forward(&self, x: Tensor<B, 4>, layer: usize) -> Tensor<B, 4>;

    fn fid_forward_layers(&self, x: Tensor<B, 4>, layers: &HashSet<usize>) -> Vec<Tensor<B, 4>>;
}

impl<B: Backend> FidForward<B> for InceptionV3<B> {
    fn fid_forward(&self, x: Tensor<B, 4>, layer: usize) -> Tensor<B, 4> {
        let n = x.shape().dims[0];

        let mut i = 0;

        macro_rules! ret_if_at_else_inc {
            ($i:ident, $layer:ident, $x:ident) => {
                if $i == $layer {
                    return $x;
                } else {
                    $i + 1
                }
            };
        }

        debug_assert_eq!(x.shape().dims, [n, 3, 299, 299]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.conv2d_1a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 32, 149, 149]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.conv2d_2a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 32, 147, 147]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.conv2d_2b_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 64, 147, 147]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.maxpool1.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 64, 73, 73]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.conv2d_3b_1x1.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 80, 73, 73]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.conv2d_4a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 192, 71, 71]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.maxpool2.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 192, 35, 35]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_5b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 256, 35, 35]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_5c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_5d.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_6a.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_6b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_6c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_6d.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_6e.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_7a.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 1280, 8, 8]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_7b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.mixed_7c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.avgpool.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        i = ret_if_at_else_inc!(i, layer, x);

        let x = self.dropout.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        ret_if_at_else_inc!(i, layer, x);

        panic!("layer {layer} out of bounds");
    }

    fn fid_forward_layers(&self, x: Tensor<B, 4>, layers: &HashSet<usize>) -> Vec<Tensor<B, 4>> {
        let mut out = Vec::new();
        let n = x.shape().dims[0];

        let mut i = 0;

        fn push_and_inc<B: Backend>(
            layers: &HashSet<usize>,
            out: &mut Vec<Tensor<B, 4>>,
            x: Tensor<B, 4>,
            i: usize,
        ) -> usize {
            if layers.contains(&i) {
                out.push(x);
            }
            i + 1
        }

        debug_assert_eq!(x.shape().dims, [n, 3, 299, 299]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.conv2d_1a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 32, 149, 149]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.conv2d_2a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 32, 147, 147]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.conv2d_2b_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 64, 147, 147]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.maxpool1.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 64, 73, 73]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.conv2d_3b_1x1.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 80, 73, 73]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.conv2d_4a_3x3.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 192, 71, 71]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.maxpool2.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 192, 35, 35]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_5b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 256, 35, 35]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_5c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_5d.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 288, 35, 35]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_6a.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_6b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_6c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_6d.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_6e.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 768, 17, 17]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_7a.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 1280, 8, 8]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_7b.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.mixed_7c.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 8, 8]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.avgpool.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        i = push_and_inc(&layers, &mut out, x.clone(), i);

        let x = self.dropout.forward(x);
        debug_assert_eq!(x.shape().dims, [n, 2048, 1, 1]);
        push_and_inc(&layers, &mut out, x.clone(), i);

        out
    }
}
