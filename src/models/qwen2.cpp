#include "models.h"

llm_build_qwen2::llm_build_qwen2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // IntelNav layer-range extensions. See llama-graph.h for field docs.
    const int32_t il_start = params.layer_start;
    const int32_t il_end   = params.layer_end >= 0
                                 ? params.layer_end
                                 : (int32_t) n_layer;
    const bool own_last_layer = (il_end == (int32_t) n_layer);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // Embed-only path: skip positions/attention/out_ids inputs, no layers,
    // no head. Every other registered input would be allocated but never
    // used by the computed graph, which the scheduler treats as an error.
    if (il_end == il_start && !params.run_head) {
        cb(inpL, "result_embd_only", -1);
        res->t_embd = inpL;
        ggml_build_forward_expand(gf, inpL);
        return;
    }

    // Head-only path: skip pos/attn inputs, no layer loop. Apply
    // output_norm + lm_head to the caller-supplied hidden state. We
    // DO register inp_out_ids — when the caller sets batch.logits on
    // fewer than all positions (n_outputs < n_tokens), the context's
    // logits-extraction reads n_outputs * n_vocab floats from the
    // start of t_logits, so t_logits must be compacted to exactly
    // those positions. This mirrors the final-layer optimization in
    // the stock full-forward path.
    if (il_end == il_start && params.run_head) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();

        ggml_tensor * cur = inpL;
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        ggml_tensor * h = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(h, "result_norm", -1);
        res->t_embd = h;

        ggml_tensor * logits = build_lora_mm(model.output, h);
        if (model.output_b != nullptr) {
            logits = ggml_add(ctx0, logits, model.output_b);
        }
        cb(logits, "result_output", -1);
        res->t_logits = logits;

        ggml_build_forward_expand(gf, logits);
        return;
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    // Only register the output-selection input when we actually own the
    // last layer (its consumer is gated on `own_last_layer` below). For
    // middle-peer partial ranges this input would otherwise be an orphan.
    ggml_tensor * inp_out_ids = own_last_layer ? build_inp_out_ids() : nullptr;

    for (int il = il_start; il < il_end; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        // Only apply the final-layer output-selection optimization when
        // the pipeline owns the true last layer. Middle peers must emit
        // every position's hidden state (inp_out_ids is nullptr there).
        if (il == (int32_t) n_layer - 1 && own_last_layer && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Partial-range peer: return pre-norm post-layer hidden state, skip
    // norm + head. The final output_norm is owned by the head_only call
    // at the end of the chain — applying it here would double-norm.
    if (!own_last_layer) {
        cb(cur, "result_embd_partial", -1);
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    // Last peer without head: same semantics as the middle peer — expose
    // pre-norm hidden state, let head_only apply norm + lm_head downstream.
    if (!params.run_head) {
        cb(cur, "result_embd_prehead", -1);
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
