/* Tiny C++ shim around the IntelNav-patched libllama.
 *
 * Purpose: the Rust FFI needs to construct llama_model_params and
 * llama_context_params, mutate a couple of fields, and pass them by
 * value to the model/context constructors. Representing those structs
 * on the Rust side is fragile — their layout evolves across llama.cpp
 * versions and any mirror would break silently on a rebase. This
 * shim handles struct construction C++-side and exposes simple
 * scalar-only functions Rust calls via dlopen.
 *
 * Built into libllama (see src/CMakeLists.txt) so distribution of
 * the IntelNav runtime is one shared library plus ggml, not three.
 */

#include "llama.h"

#include <atomic>
#include <cstdint>

namespace {

/* Set via intelnav_trip_abort() from Rust. Read from abort_cb(),
 * which libllama calls periodically from the compute loop. Returning
 * `true` from the callback tells ggml to abort the current graph
 * cleanly (via its error return path), instead of the default exit(1)
 * that fires from ggml_abort() on fatal asserts.
 *
 * This is the fix for a DoS vector: without a registered callback a
 * crafted ForwardHidden payload that trips a ggml assertion takes
 * down the host process. With the callback + a tripped flag, libllama
 * returns GGML_STATUS_ABORTED and we propagate as a normal Rust error.
 */
std::atomic<bool> g_abort_flag{false};

bool abort_cb(void * /*data*/) {
    return g_abort_flag.load(std::memory_order_relaxed);
}

} // namespace

extern "C" {

struct llama_model * intelnav_load_model(
        const char * path_model,
        int32_t      n_gpu_layers) {
    llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path_model, p);
}

struct llama_context * intelnav_new_context(
        struct llama_model * model,
        uint32_t             n_ctx,
        uint32_t             n_batch,
        uint32_t             n_ubatch,
        uint32_t             n_seq_max) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_batch;
    cp.n_ubatch  = n_ubatch;
    cp.no_perf   = true;
    /* Caller decides how many seq slots it needs. One is the layer
     * loop's real KV (seq 0); one is the scratch seq the Rust adapter
     * uses for head_only/embed_only (seq 1); the rest are room for
     * M3's continuous-batching gateway. 2 is the current single-
     * session runtime's minimum; gateway deployments will pass 16+. */
    cp.n_seq_max = n_seq_max;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx != nullptr) {
        /* Install the abort callback so a crafted payload can't take
         * the host process down via ggml_abort(). Rust flips the flag
         * via intelnav_trip_abort(). */
        llama_set_abort_callback(ctx, abort_cb, nullptr);
    }
    return ctx;
}

void intelnav_trip_abort(bool tripped) {
    g_abort_flag.store(tripped, std::memory_order_relaxed);
}

} /* extern "C" */
