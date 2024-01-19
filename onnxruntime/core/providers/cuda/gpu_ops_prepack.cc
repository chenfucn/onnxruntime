// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#ifdef USE_CUDA

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

#include "blk_q4/f16_prepack_sm80.h"

namespace onnxruntime {

class CUDAExecutionProvider;

/**
@Class AttentionFusion
Rewrite graph fusing attention subgraph to a single Attention node.
*/
class GpuOpsPrepack : public GraphTransformer {
 public:
  GpuOpsPrepack(const CUDAExecutionProvider& cuda_ep) noexcept
      : GraphTransformer("GpuOpsPrepack", InlinedHashSet<std::string_view>{onnxruntime::kCudaExecutionProvider}),
        cuda_ep_(cuda_ep) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  const CUDAExecutionProvider& cuda_ep_;
};


GraphTransformer* CreateGpuOpsPrepack(const CUDAExecutionProvider& cuda_ep) {
  return new GpuOpsPrepack(cuda_ep);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Operator specific logic for
// 1. whether prepacking is supported.
// 2. actually rewrite the initializers.
//
// Operator prepacking logic should be placed close to the operator implementation.
// instead of putting them all here. Need to figure out how to do it.


template <typename T>
const gsl::span<T const> ArgToDataSpan(const NodeArg* arg, const Graph& graph) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_ENFORCE(graph.GetInitializedTensor(arg->Name(), tensor_proto),
              "Missing initializer for ", arg->Name());
  Initializer initializer(*tensor_proto, graph.ModelPath());
  return initializer.DataAsSpan<T>();
}

namespace contrib {
namespace cuda {

/**
 *@brief Check if the operator is a MatMulNBits<float16> and it can be prepacked.
 */
extern bool should_pack_matmul_nbits(const Node& node, const CUDAExecutionProvider& cuda_ep);

}  // namespace cuda
}  // namespace contrib

Status GpuOpsPrepack::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.GetExecutionProviderType() != onnxruntime::kCudaExecutionProvider) {
      continue;  // we can only care nodes on CUDA EPs
    }

    if (contrib::cuda::should_pack_matmul_nbits(node, cuda_ep_)){
      // Still MatMulNBits<float16> specific code, should not be here.
      auto& node_name = node.Name();
      auto& mutable_input_defs = node.MutableInputDefs();

      NodeArg* old_weights_arg = mutable_input_defs[1];
      const gsl::span<uint8_t const> weights = ArgToDataSpan<uint8_t>(old_weights_arg, graph);

      ONNX_NAMESPACE::TensorProto initializer;
      initializer.set_name(graph.GenerateNodeArgName(node_name + "prepacked_weight"));
      initializer.add_dims(weights.size());
      initializer.set_data_type(onnxruntime::utils::ToTensorProtoElementType<uint8_t>());
      initializer.set_raw_data(weights.data(), weights.size_bytes());
      NodeArg& result = graph_utils::AddInitializer(graph, initializer);


      // pack weights here

      graph.RemoveConsumerNode(old_weights_arg->Name(), &node);
      mutable_input_defs[1] = &result;
      graph.AddConsumerNode(result.Name(), &node);
    }

  }

  return Status::OK();
}


}  // namespace onnxruntime

//#endif  // USE_CUDA
