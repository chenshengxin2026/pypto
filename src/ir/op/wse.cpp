/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/ir/wse.h"

#include <any>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {
TypePtr DeduceGather(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs, bool tile) {
  CHECK(args.size() == 1) << "WSE allgather expects one input";
  CHECK(tile || !As<DistributedTensorType>(args[0]->GetType()))
      << "WSE adapter does not bind DistributedTensorType operands; use a local Tensor";
  std::shared_ptr<const ShapedType> input =
      tile ? std::static_pointer_cast<const ShapedType>(As<TileType>(args[0]->GetType()))
           : std::static_pointer_cast<const ShapedType>(As<TensorType>(args[0]->GetType()));
  CHECK(input && input->shape_.size() == 2 && input->dtype_ == DataType::FP16)
      << "WSE allgather requires a rank-2 FP16 input";
  std::vector<int> members;
  int axis = -1;
  for (const auto& [key, value] : kwargs) {
    if (key == "members") members = ParseWseMembers(AnyCast<std::string>(value, key));
    if (key == "axis") axis = AnyCast<int>(value, key);
  }
  CHECK(!members.empty() && members.size() <= 24) << "A3 WSE requires 1..24 resident members";
  CHECK(std::set<int>(members.begin(), members.end()).size() == members.size())
      << "WSE group members must be distinct";
  for (auto member : members) CHECK(member >= 0 && member <= INT32_MAX) << "Invalid logical block ID";
  CHECK(axis == 0 || axis == 1) << "WSE allgather axis must be 0 or 1";
  auto shape = input->shape_;
  for (const auto& dim : shape) {
    auto value = As<ConstInt>(dim);
    CHECK(value && value->value_ > 0 && value->value_ <= 65536)
        << "WSE allgather requires static dimensions in 1..65536";
  }
  CHECK(As<ConstInt>(shape[1])->value_ % 16 == 0) << "A3 WSE row bytes must be aligned to 32";
  int64_t extent = As<ConstInt>(shape[axis])->value_ * members.size();
  shape[axis] = std::make_shared<ConstInt>(extent, DataType::INDEX, args[0]->span_);
  int64_t bytes = As<ConstInt>(shape[0])->value_ * As<ConstInt>(shape[1])->value_ * 2;
  CHECK(bytes <= 128 * 1024) << "A3 WSE receive tile exceeds 128 KiB";
  if (!tile) return std::make_shared<TensorType>(shape, input->dtype_);
  TileView view;
  view.blayout = TileLayout::row_major;
  view.slayout = TileLayout::none_box;
  view.valid_shape = shape;
  return std::make_shared<TileType>(shape, input->dtype_, std::nullopt, view, MemorySpace::Mat);
}
}  // namespace

REGISTER_OP("tensor.wse_allgather")
    .set_op_category("TensorOp")
    .set_description("Synchronously gather FP16 shards in ordered logical WSE group order")
    .add_argument("input", "Local tensor shard")
    .set_attr<std::string>("members")
    .set_attr<int>("axis")
    .f_deduce_type([](const auto& args, const auto& kwargs) { return DeduceGather(args, kwargs, false); });

REGISTER_OP("tile.wse_allgather")
    .set_op_category("TileOp")
    .set_description("Synchronously exchange flat L1 shards using the WSE Calendar contract")
    .functional_execution_memory_access()
    .set_core_affinity(core_affinity::CoreAffinity::CUBE)
    .add_argument("input", "Flat L1 send tile")
    .set_attr<std::string>("members")
    .set_attr<int>("axis")
    .set_attr<int>("exp_bytes")
    .set_input_memory(0, MemorySpace::Mat)
    .set_output_memory(MemorySpace::Mat)
    .forbid_output_alias(0)
    .f_deduce_type([](const auto& args, const auto& kwargs) { return DeduceGather(args, kwargs, true); });
}  // namespace pypto::ir
