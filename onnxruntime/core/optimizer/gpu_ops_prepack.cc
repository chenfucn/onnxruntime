// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#ifdef USE_CUDA

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/gpu_ops_prepack.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

#include "blk_q4/f16_prepack_sm80.h"

#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {

extern ProviderInfo_CUDA* TryGetProviderInfo_CUDA();

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

/**
 *@brief Check if the operator is a MatMulNBits<float16> and it can be prepacked.
 */
bool should_pack_matmul_nbits(const Node& node) {

  if (node.GetExecutionProviderType() != onnxruntime::kCudaExecutionProvider) {
    return false;  // unknown provider
  }

  //
  // If we want to expand this prepacking logic to other operators, we need a lookup
  // mechanism to find:
  // 1. whether the operator supports prepacking.
  // 2. the operator specific prepacking logic.
  //
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMulNBits", {1}, kMSDomain)) {
    return false;
  }

  const auto* acts = node.InputDefs()[0];

  if (acts == nullptr || acts->Type() == nullptr || acts->Type()->find("float16") == std::string::npos) {
    return false;
  }

  int k = node.GetAttributes().at("K").i();
  int n = node.GetAttributes().at("N").i();
  int block_size = node.GetAttributes().at("block_size").i();
  int nbits = node.GetAttributes().at("bits").i();

  if (nbits != 4) {
    return false;
  }

  auto* provider_info = TryGetProviderInfo_CUDA();
  ORT_ENFORCE(provider_info != nullptr, "Failed to query CUDA provider info while prepacking cuda operators.");
  int major, minor;
  ORT_ENFORCE(provider_info->GetCurrentGpuDeviceVersion(&major, &minor) == nullptr,
             "Failed to query CUDA device version while prepacking cuda operators.");

  switch (block_size)
  {
  case 16:
    return onnxruntime::cuda::BlkQuantGemmSm80Supported<MLFloat16, 16, 4>(k, n, major, minor);
  case 32:
    return onnxruntime::cuda::BlkQuantGemmSm80Supported<MLFloat16, 32, 4>(k, n, major, minor);
  case 64:
    return onnxruntime::cuda::BlkQuantGemmSm80Supported<MLFloat16, 64, 4>(k, n, major, minor);
  }
  return false;
}


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

    if (!should_pack_matmul_nbits(node)){
      continue;
    }

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

  return Status::OK();
}


}  // namespace onnxruntime

//#endif  // USE_CUDA
