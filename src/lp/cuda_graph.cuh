#pragma once

#ifdef MIPX_HAS_CUDA

#include <cuda_runtime.h>

namespace mipx {
namespace gpu {

// Single CUDA Graph for the PDLP iteration loop.
// No pointer swapping occurs between iterations, so a single graph suffices.
class IterationGraph {
public:
    IterationGraph() = default;

    ~IterationGraph() {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
    }

    IterationGraph(const IterationGraph&) = delete;
    IterationGraph& operator=(const IterationGraph&) = delete;

    bool hasGraph() const { return init_; }

    bool beginCapture(cudaStream_t stream) {
        if (init_) return false;
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) return false;
        capturing_ = true;
        return true;
    }

    bool endCapture(cudaStream_t stream) {
        if (!capturing_) return false;
        capturing_ = false;

        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || !graph) {
            return false;
        }

        cudaGraphExec_t exec = nullptr;
        if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
            cudaGraphDestroy(graph);
            return false;
        }

        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
        graph_ = graph;
        exec_ = exec;
        init_ = true;
        return true;
    }

    bool launch(cudaStream_t stream) {
        if (!exec_) return false;
        return cudaGraphLaunch(exec_, stream) == cudaSuccess;
    }

    void invalidate() {
        if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
        if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        init_ = false;
    }

    bool isCapturing() const { return capturing_; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    bool init_ = false;
    bool capturing_ = false;
};

}  // namespace gpu
}  // namespace mipx

#endif  // MIPX_HAS_CUDA
