// Finite-difference / identity checks for the pointer-generator gradient path.
//
// The audit warns that once cross-entropy is computed on the mixture P_final (not a raw
// softmax), the fused (pred - target) shortcut is invalid: you must use the true
// CE-on-probability gradient and the real softmax Jacobian. This harness validates exactly
// that, in isolation, so a sign/factor error can't hide inside a full training run.
//
// Build: make MAIN_SRC=examples/gradcheck_pointer.cpp && ./gradcheck_pointer

#include "../include/core/tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Kernels under test (defined in src/core/cuda_ops.cu)
void full_attention_softmax_cuda(float* data, int rows, int cols);
void full_attention_softmax_backward_cuda(const float* dY, const float* Y, float* dX, int rows, int cols);
void ce_prob_grad_cuda(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens);
void blend_forward_cuda(const float* p_gen, const float* P_vocab, const float* P_schema, float* P_final, int batch_seq, int V);
void blend_backward_dist_cuda(const float* p_gen, const float* dP_final, float* dP_vocab, float* dP_schema, int batch_seq, int V);
void blend_backward_pgen_cuda(const float* P_vocab, const float* P_schema, const float* dP_final, float* dp_gen, int batch_seq, int V);

static const float EPS_LS = 0.1f; // label smoothing (must match the kernels)

static float smoothed_target(int v, int target, int V) {
    return (v == target) ? (1.0f - EPS_LS) + (EPS_LS / V) : (EPS_LS / V);
}

// CE loss on a probability distribution row (label smoothed, /valid).
static double ce_loss_row(const std::vector<float>& P, int row, int V, int target, int valid) {
    if (target == -100) return 0.0;
    double L = 0.0;
    for (int v = 0; v < V; ++v)
        L += -smoothed_target(v, target, V) * std::log(P[row * V + v] + 1e-7f);
    return L / valid;
}

int main() {
    int N = 3, V = 6, valid = 2;
    std::vector<int> targets = {2, 4, -100};

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---------- Test 1: ce_prob_grad ∘ softmax_Jacobian == fused (P - t)/valid ----------
    std::vector<float> logits(N * V);
    for (auto& x : logits) x = nd(rng);

    Tensor zP = Tensor::upload(logits, {N, V});
    full_attention_softmax_cuda(zP.data(), N, V);   // in-place softmax
    std::vector<float> P = zP.download();

    std::vector<float> tgt_f(N);
    for (int i = 0; i < N; ++i) tgt_f[i] = (float)targets[i];
    Tensor dtgt = Tensor::upload(tgt_f, {N, 1});

    Tensor dP(std::vector<int>{N, V});
    ce_prob_grad_cuda(zP.data(), dtgt.data(), dP.data(), N, V, valid);

    Tensor dZ(std::vector<int>{N, V});
    full_attention_softmax_backward_cuda(dP.data(), zP.data(), dZ.data(), N, V);
    std::vector<float> dz = dZ.download();

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int v = 0; v < V; ++v) {
            float fused = (targets[i] == -100) ? 0.0f
                        : (P[i * V + v] - smoothed_target(v, targets[i], V)) / valid;
            max_err = std::max(max_err, (double)std::fabs(dz[i * V + v] - fused));
        }
    }
    std::cout << "[T1] softmax-Jacobian(ce_prob) vs fused (P-t)/valid : max_err = " << max_err << "\n";

    // ---------- Test 2: ce_prob_grad matches finite-difference of CE loss on P_final ----------
    // Build a valid mixture distribution and check dL/dP_final entrywise.
    std::vector<float> pv(N * V), ps(N * V), pg(N);
    for (int i = 0; i < N; ++i) {
        float sv = 0, ss = 0;
        for (int v = 0; v < V; ++v) { pv[i*V+v] = std::fabs(nd(rng)) + 0.1f; sv += pv[i*V+v]; }
        for (int v = 0; v < V; ++v) { ps[i*V+v] = std::fabs(nd(rng)) + 0.1f; ss += ps[i*V+v]; }
        for (int v = 0; v < V; ++v) { pv[i*V+v] /= sv; ps[i*V+v] /= ss; }
        pg[i] = 0.3f + 0.4f * (float)(i) / N; // in (0,1)
    }
    Tensor Pv = Tensor::upload(pv, {N, V});
    Tensor Ps = Tensor::upload(ps, {N, V});
    Tensor Pg = Tensor::upload(pg, {N, 1});
    Tensor Pf(std::vector<int>{N, V});
    blend_forward_cuda(Pg.data(), Pv.data(), Ps.data(), Pf.data(), N, V);
    std::vector<float> Pfinal = Pf.download();

    Tensor dPf(std::vector<int>{N, V});
    ce_prob_grad_cuda(Pf.data(), dtgt.data(), dPf.data(), N, V, valid);
    std::vector<float> dpf = dPf.download();

    double fd_err = 0.0;
    float h = 1e-3f;
    for (int i = 0; i < N; ++i) {
        for (int v = 0; v < V; ++v) {
            std::vector<float> Pp = Pfinal, Pm = Pfinal;
            Pp[i*V+v] += h; Pm[i*V+v] -= h;
            double Lp = ce_loss_row(Pp, i, V, targets[i], valid);
            double Lm = ce_loss_row(Pm, i, V, targets[i], valid);
            double num = (Lp - Lm) / (2 * h);
            fd_err = std::max(fd_err, std::fabs(num - dpf[i*V+v]));
        }
    }
    std::cout << "[T2] ce_prob_grad vs finite-diff dL/dP_final        : max_err = " << fd_err << "\n";

    // ---------- Test 3: blend backward splits + p_gen reduction ----------
    Tensor dPvb(std::vector<int>{N, V}), dPsb(std::vector<int>{N, V}), dpg(std::vector<int>{N, 1});
    blend_backward_dist_cuda(Pg.data(), dPf.data(), dPvb.data(), dPsb.data(), N, V);
    blend_backward_pgen_cuda(Pv.data(), Ps.data(), dPf.data(), dpg.data(), N, V);
    std::vector<float> dPvb_h = dPvb.download(), dPsb_h = dPsb.download(), dpg_h = dpg.download();

    double blend_err = 0.0, pgen_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int v = 0; v < V; ++v) {
            blend_err = std::max(blend_err, (double)std::fabs(dPvb_h[i*V+v] - pg[i]*dpf[i*V+v]));
            blend_err = std::max(blend_err, (double)std::fabs(dPsb_h[i*V+v] - (1.0f-pg[i])*dpf[i*V+v]));
            s += (pv[i*V+v] - ps[i*V+v]) * dpf[i*V+v];
        }
        pgen_err = std::max(pgen_err, std::fabs(s - dpg_h[i]));
    }
    std::cout << "[T3] blend backward dist split                      : max_err = " << blend_err << "\n";
    std::cout << "[T3] blend backward p_gen reduction                 : max_err = " << pgen_err << "\n";

    double tol = 2e-3;
    bool ok = (max_err < tol) && (fd_err < tol) && (blend_err < 1e-5) && (pgen_err < 1e-4);
    std::cout << (ok ? "\nGRADCHECK PASSED\n" : "\nGRADCHECK FAILED\n");
    return ok ? 0 : 1;
}
