#pragma once
#include "layer.h"
#include <fstream>
#include "../core/tensor.h"
#include "self_attention.h"
#include "pointer_attention.h"
#include "layernorm.h"
#include "dense.h"
#include "activation.h"
#include "dropout.h"

// DecoderBlock: C++ port of scripts/schema_fusion_pt.py::DecoderBlock.
// Pre-LN: self-attn (causal) -> cross-attn(memory) -> FFN, each with dropout
// on the residual branch:
//   h = ln1(x); x = x + drop(self_attn(h))
//   h = ln2(x); x = x + drop(cross_attn(h, mem))
//   h = ln3(x); x = x + drop(ff2(relu(ff1(h))))
// The cross-attention REUSES PointerAttention purely as a context builder;
// its attention weights are NOT the pointer distribution (the model has a
// dedicated pointer head at the top -- see deep_fusion_net.h).
class DecoderBlock
{
    private:
        int dimension;

        LayerNorm ln1, ln2, ln3;
        SelfAttention self_attn;     // causal
        PointerAttention cross_attn; // context over the memory bank
        Dense d1, d2;
        Activation act;
        Dropout drop1, drop2, drop3;

        Tensor cached_input; // for shapes in backward

    public:
        DecoderBlock(int dimension, int heads, float dropout)
            : dimension(dimension),
              ln1(dimension), ln2(dimension), ln3(dimension),
              self_attn(dimension, heads, /*causal=*/true),
              cross_attn(dimension, heads),
              d1(dimension, dimension * 4), d2(dimension * 4, dimension),
              act(ActivationType::RELU, 0.0f),
              drop1(dropout), drop2(dropout), drop3(dropout)
        {}

        // x: [batch, seq, dim]; mem: [batch, M, dim]; mem_mask: [batch, M]
        Tensor forward(const Tensor& x, const Tensor& mem, const Tensor& mem_mask)
        {
            cached_input = x;
            int n = x.shape[0], t = x.shape[1], d = x.shape[2];

            // -- self-attention sublayer --
            Tensor h1 = ln1.forward(x);
            Tensor a1 = self_attn.forward(h1);
            a1 = drop1.forward(a1);
            Tensor x1 = x + a1;

            // -- cross-attention sublayer (context only) --
            Tensor h2 = ln2.forward(x1);
            auto out = cross_attn.forward_dual(h2, mem, mem_mask);
            Tensor ctx = drop2.forward(out.first);
            Tensor x2 = x1 + ctx;

            // -- FFN sublayer --
            Tensor h3 = ln3.forward(x2).reshape({n * t, d});
            Tensor f = d1.forward(h3);
            f = act.forward(f);
            f = d2.forward(f);
            f = drop3.forward(f);
            Tensor x3 = x2 + f.reshape({n, t, d});

            return x3;
        }

        // Returns dx. The memory gradient (summed over the cross-attn path)
        // is retrievable afterwards via get_mem_grad() -> [batch, M, dim].
        Tensor backward(const Tensor& dY, float lr)
        {
            int n = cached_input.shape[0], t = cached_input.shape[1], d = cached_input.shape[2];

            // -- FFN sublayer backward --
            Tensor df = dY.reshape({n * t, d});
            df = drop3.backward(df, lr);
            df = d2.backward(df, lr);
            df = act.backward(df, lr);
            df = d1.backward(df, lr);
            Tensor dh3 = ln3.backward(df.reshape({n, t, d}), lr);
            Tensor dx2 = dh3 + dY;

            // -- cross-attention sublayer backward --
            Tensor dctx = drop2.backward(dx2, lr);
            Tensor dh2 = cross_attn.backward_ext(dctx, lr, nullptr);
            dh2.shape = {n, t, d};
            Tensor dx1 = ln2.backward(dh2, lr) + dx2;

            // -- self-attention sublayer backward --
            Tensor da1 = drop1.backward(dx1, lr);
            Tensor dh1 = self_attn.backward(da1, lr);
            Tensor dx = ln1.backward(dh1, lr) + dx1;

            return dx;
        }

        Tensor get_mem_grad() const { return cross_attn.get_schema_grad(); }

        void set_mode(bool train)
        {
            ln1.set_mode(train); ln2.set_mode(train); ln3.set_mode(train);
            self_attn.set_mode(train); cross_attn.set_mode(train);
            d1.set_mode(train); d2.set_mode(train); act.set_mode(train);
            drop1.set_mode(train); drop2.set_mode(train); drop3.set_mode(train);
        }

        void save(std::ofstream& os)
        {
            ln1.save(os); ln2.save(os); ln3.save(os);
            self_attn.save(os); cross_attn.save(os);
            d1.save(os); d2.save(os);
        }

        void load(std::ifstream& is)
        {
            ln1.load(is); ln2.load(is); ln3.load(is);
            self_attn.load(is); cross_attn.load(is);
            d1.load(is); d2.load(is);
        }
};
