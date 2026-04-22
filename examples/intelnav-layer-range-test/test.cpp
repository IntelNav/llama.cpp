// IntelNav layer-range test
//
// Proves that the patched llama.cpp produces bit-identical (or within
// fp-rounding) logits when a forward pass is composed piecewise from
// the three new public functions:
//
//   ref:       llama_decode(tokens)                             -> logits_ref
//   piece A:   embed_only(tokens)
//              decode_layers(hidden, 0, n_layer)
//              head_only(hidden)                                -> logits_new
//   piece B:   decode_layers(tokens, 0, n_layer/2)
//              decode_layers(hidden_mid, n_layer/2, n_layer)
//              head_only(hidden_tail)                           -> logits_two
//
// Piece A exercises all three new functions. Piece B mirrors what an
// actual IntelNav 2-peer pipeline does on the wire: the first peer
// owns embed + layers [0, N/2); the second peer owns layers [N/2, N)
// + head. If piece B matches the reference, arbitrary N-peer splits
// (which are just more applications of the same pattern) will too.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct diff_report {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int   argmax_ref = -1;
    int   argmax_new = -1;
    float v_ref = -1e30f;
    float v_new = -1e30f;
};

static diff_report compare_logits(const std::vector<float> & a, const std::vector<float> & b) {
    diff_report d;
    const int n = (int) a.size();
    for (int i = 0; i < n; ++i) {
        const float ad = std::fabs(a[i] - b[i]);
        if (ad > d.max_abs) d.max_abs = ad;
        if (std::fabs(a[i]) > 1e-6f) {
            const float rd = ad / std::fabs(a[i]);
            if (rd > d.max_rel) d.max_rel = rd;
        }
        if (a[i] > d.v_ref) { d.v_ref = a[i]; d.argmax_ref = i; }
        if (b[i] > d.v_new) { d.v_new = b[i]; d.argmax_new = i; }
    }
    return d;
}

static void print_diff(const char * label, const diff_report & d) {
    printf("  %s\n", label);
    printf("    max_abs_diff = %.6e\n", d.max_abs);
    printf("    max_rel_diff = %.6e\n", d.max_rel);
    printf("    argmax  ref=%d (%+f)  new=%d (%+f)  %s\n",
            d.argmax_ref, d.v_ref, d.argmax_new, d.v_new,
            d.argmax_ref == d.argmax_new ? "MATCH" : "DIVERGE");
}

static void fill_tokens_batch(
        llama_batch & b,
        const std::vector<llama_token> & tokens,
        bool logits_all) {
    const int n = (int) tokens.size();
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i] = tokens[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = logits_all ? 1 : (i == n - 1 ? 1 : 0);
    }
}

static void fill_embd_batch(
        llama_batch & b,
        const std::vector<float> & hidden,
        int n_tokens,
        int n_embd,
        bool logits_last_only) {
    b.n_tokens = n_tokens;
    std::memcpy(b.embd, hidden.data(), (size_t) n_tokens * n_embd * sizeof(float));
    for (int i = 0; i < n_tokens; ++i) {
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = logits_last_only ? (i == n_tokens - 1 ? 1 : 0) : 1;
    }
}

static bool pull_hidden(
        llama_context * ctx,
        int n_tokens,
        int n_embd,
        std::vector<float> & out) {
    out.assign((size_t) n_tokens * n_embd, 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        float * e = llama_get_embeddings_ith(ctx, i);
        if (!e) {
            fprintf(stderr, "  get_embeddings_ith(%d) returned null\n", i);
            return false;
        }
        std::memcpy(&out[(size_t) i * n_embd], e, (size_t) n_embd * sizeof(float));
    }
    return true;
}

static void reset_seq(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_seq_rm(mem, 0, -1, -1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = argc >= 3 ? argv[2] : "Hello my name is";

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU for deterministic fp32 math
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_embd  = llama_model_n_embd(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t       n_layer = llama_model_n_layer(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                         nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                   tokens.data(), tokens.size(), true, true);

    printf("model: %s\n", model_path);
    printf("prompt: \"%s\"\n", prompt.c_str());
    printf("n_tokens=%d  n_embd=%d  n_vocab=%d  n_layer=%d\n\n",
            n_prompt, n_embd, n_vocab, n_layer);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 512;
    cp.n_batch = 256;
    cp.no_perf = true;

    // ---------- reference ----------
    std::vector<float> logits_ref(n_vocab);
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch batch = llama_batch_init(n_prompt, 0, 1); // tokens path
        fill_tokens_batch(batch, tokens, /*logits_all=*/false);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "reference decode failed\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits from reference\n"); return 1; }
        std::memcpy(logits_ref.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }
    printf("reference: llama_decode done\n");

    // ---------- test 0: decode_layers(0, N, run_head=true) on tokens ----------
    // Should be bit-identical to llama_decode — strips embed_only and head_only
    // out of the equation to isolate any bug in the core layer-range patch.
    std::vector<float> logits_0(n_vocab);
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch batch = llama_batch_init(n_prompt, 0, 1); // tokens path
        fill_tokens_batch(batch, tokens, /*logits_all=*/false);
        if (llama_decode_layers(ctx, batch, 0, n_layer, /*run_head=*/true) != 0) {
            fprintf(stderr, "test 0 decode_layers(0, N, run_head=true) failed\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits in test 0\n"); return 1; }
        std::memcpy(logits_0.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }
    printf("test 0:    decode_layers(tokens, 0, N, run_head=true) done\n");

    // ---------- test 0.5: embed_only + decode_layers(embd, 0, N, run_head=true) ----------
    // If test 0 matches ref but this diverges, the bug is in either
    // embed_only or the embd-input path of decode_layers (not in head_only).
    std::vector<float> logits_half(n_vocab);
    {
        llama_context * ctx = llama_init_from_model(model, cp);

        llama_batch b1 = llama_batch_init(n_prompt, 0, 1);
        fill_tokens_batch(b1, tokens, /*logits_all=*/true);
        if (llama_embed_only(ctx, b1) != 0) {
            fprintf(stderr, "test 0.5 embed_only failed\n"); return 1;
        }
        std::vector<float> hidden_embd;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_embd)) return 1;
        llama_batch_free(b1);
        reset_seq(ctx);

        llama_batch b2 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b2, hidden_embd, n_prompt, n_embd, /*logits_last_only=*/false);
        if (llama_decode_layers(ctx, b2, 0, n_layer, /*run_head=*/true) != 0) {
            fprintf(stderr, "test 0.5 decode_layers(embd, 0, N, run_head=true) failed\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits in test 0.5\n"); return 1; }
        std::memcpy(logits_half.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(b2);
        llama_free(ctx);
    }
    printf("test 0.5:  embed_only + decode_layers(embd, 0, N, run_head=true) done\n");

    // ---------- test 0.75: decode_layers(tokens, 0, N, run_head=false) + head_only ----------
    // Isolates head_only from embed_only: uses decode_layers with run_head=false
    // to produce the known-good post-layer pre-norm hidden state (same state as
    // test 0's inner computation before the head), then applies head_only on it.
    // If this diverges, the bug is in head_only (and not in embed_only).
    std::vector<float> logits_head(n_vocab);
    {
        llama_context * ctx = llama_init_from_model(model, cp);

        llama_batch b1 = llama_batch_init(n_prompt, 0, 1);
        fill_tokens_batch(b1, tokens, /*logits_all=*/true);
        if (llama_decode_layers(ctx, b1, 0, n_layer, /*run_head=*/false) != 0) {
            fprintf(stderr, "test 0.75 decode_layers(tokens, 0, N, run_head=false) failed\n"); return 1;
        }
        std::vector<float> hidden_prenorm;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_prenorm)) return 1;
        llama_batch_free(b1);
        reset_seq(ctx);

        llama_batch b2 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b2, hidden_prenorm, n_prompt, n_embd, /*logits_last_only=*/true);
        if (llama_head_only(ctx, b2) != 0) {
            fprintf(stderr, "test 0.75 head_only failed\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits in test 0.75\n"); return 1; }
        std::memcpy(logits_head.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(b2);
        llama_free(ctx);
    }
    printf("test 0.75: decode_layers(tokens, 0, N, run_head=false) + head_only done\n");

    // ---------- piecewise A: embed + full decode_layers + head ----------
    std::vector<float> logits_A(n_vocab);
    {
        llama_context * ctx = llama_init_from_model(model, cp);

        // step 1: tokens -> embed_only -> hidden_embd
        llama_batch b1 = llama_batch_init(n_prompt, 0, 1);
        fill_tokens_batch(b1, tokens, /*logits_all=*/true);
        if (llama_embed_only(ctx, b1) != 0) {
            fprintf(stderr, "embed_only failed\n"); return 1;
        }
        std::vector<float> hidden_embd;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_embd)) return 1;
        llama_batch_free(b1);
        reset_seq(ctx); // embed_only shouldn't touch KV but clear anyway

        // step 2: hidden_embd -> decode_layers(0, n_layer) -> hidden_post
        llama_batch b2 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b2, hidden_embd, n_prompt, n_embd, /*logits_last_only=*/false);
        if (llama_decode_layers(ctx, b2, 0, n_layer, /*run_head=*/false) != 0) {
            fprintf(stderr, "decode_layers(0, %d) failed\n", n_layer); return 1;
        }
        std::vector<float> hidden_post;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_post)) return 1;
        llama_batch_free(b2);
        reset_seq(ctx); // clear KV before head_only rerun of same positions

        // step 3: hidden_post -> head_only -> logits
        llama_batch b3 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b3, hidden_post, n_prompt, n_embd, /*logits_last_only=*/true);
        if (llama_head_only(ctx, b3) != 0) {
            fprintf(stderr, "head_only failed\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits from head_only\n"); return 1; }
        std::memcpy(logits_A.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(b3);
        llama_free(ctx);
    }
    printf("piece A:   embed_only + decode_layers(0, %d) + head_only done\n", n_layer);

    // ---------- piecewise B: two-peer split ----------
    std::vector<float> logits_B(n_vocab);
    const int n_mid = n_layer / 2;
    {
        llama_context * ctx = llama_init_from_model(model, cp);

        // peer 1: tokens -> decode_layers(0, n_mid) -> hidden_mid
        llama_batch b1 = llama_batch_init(n_prompt, 0, 1); // tokens path
        fill_tokens_batch(b1, tokens, /*logits_all=*/true);
        if (llama_decode_layers(ctx, b1, 0, n_mid, /*run_head=*/false) != 0) {
            fprintf(stderr, "decode_layers(0, %d) failed\n", n_mid); return 1;
        }
        std::vector<float> hidden_mid;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_mid)) return 1;
        llama_batch_free(b1);
        reset_seq(ctx); // KV from layers [0, n_mid) drops out for the next run

        // peer 2a: hidden_mid -> decode_layers(n_mid, n_layer) -> hidden_tail
        llama_batch b2 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b2, hidden_mid, n_prompt, n_embd, /*logits_last_only=*/false);
        if (llama_decode_layers(ctx, b2, n_mid, n_layer, /*run_head=*/false) != 0) {
            fprintf(stderr, "decode_layers(%d, %d) failed\n", n_mid, n_layer); return 1;
        }
        std::vector<float> hidden_tail;
        if (!pull_hidden(ctx, n_prompt, n_embd, hidden_tail)) return 1;
        llama_batch_free(b2);
        reset_seq(ctx);

        // peer 2b: hidden_tail -> head_only -> logits
        llama_batch b3 = llama_batch_init(n_prompt, n_embd, 1);
        fill_embd_batch(b3, hidden_tail, n_prompt, n_embd, /*logits_last_only=*/true);
        if (llama_head_only(ctx, b3) != 0) {
            fprintf(stderr, "head_only failed in peer B\n"); return 1;
        }
        float * l = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!l) { fprintf(stderr, "no logits from peer B head_only\n"); return 1; }
        std::memcpy(logits_B.data(), l, (size_t) n_vocab * sizeof(float));
        llama_batch_free(b3);
        llama_free(ctx);
    }
    printf("piece B:   decode_layers(0, %d) + decode_layers(%d, %d) + head_only done\n\n",
            n_mid, n_mid, n_layer);

    // ---------- compare ----------
    diff_report d0    = compare_logits(logits_ref, logits_0);
    diff_report dHalf = compare_logits(logits_ref, logits_half);
    diff_report dHead = compare_logits(logits_ref, logits_head);
    diff_report dA    = compare_logits(logits_ref, logits_A);
    diff_report dB    = compare_logits(logits_ref, logits_B);
    printf("results (fp32 CPU path, logits for token index %d):\n", n_prompt - 1);
    print_diff("[test 0]    decode_layers(tokens, 0, N, run_head=true)  vs  llama_decode", d0);
    print_diff("[test 0.5]  embed_only + decode_layers(embd, 0, N, run_head=true)  vs  llama_decode", dHalf);
    print_diff("[test 0.75] decode_layers(tokens, 0, N, run_head=false) + head_only  vs  llama_decode", dHead);
    print_diff("[test 1]    embed_only + decode_layers(0, N) + head_only  vs  llama_decode", dA);
    print_diff("[test 2]    decode_layers(0, N/2) + decode_layers(N/2, N) + head_only  vs  llama_decode", dB);

    const bool pass = (d0.argmax_ref    == d0.argmax_new)
                   && (dHalf.argmax_ref == dHalf.argmax_new)
                   && (dHead.argmax_ref == dHead.argmax_new)
                   && (dA.argmax_ref    == dA.argmax_new)
                   && (dB.argmax_ref    == dB.argmax_new);

    llama_model_free(model);

    printf("\nsummary: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
