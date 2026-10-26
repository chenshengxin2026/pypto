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
#include <algorithm>
#include <memory>
#include <string>

#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/wse.h"

namespace pypto::ir {
namespace {
class FinalizeCalendar : public IRMutator {
 public:
  ExprPtr VisitExpr_(const CallPtr& call) override {
    if (!IsOp(call, "tile.wse_allgather")) return IRMutator::VisitExpr_(call);
    CHECK_SPAN(++count_ == 1 && control_depth_ == 0, call->span_)
        << "A3 WSE requires one unconditional Calendar call per kernel; split stages into kernels";
    auto* context = PassContext::Current();
    CHECK_SPAN(!context || context->GetMemoryPlanner() == MemoryPlanner::PyPTO, call->span_)
        << "WSE Calendar currently requires MemoryPlanner.PYPTO";
    const auto send = As<TileType>(call->args_[0]->GetType());
    const auto recv = As<TileType>(result_type_);
    for (const auto& tile : {send, recv}) {
      CHECK_SPAN(tile && tile->GetMemorySpace() == MemorySpace::Mat && tile->memref_.has_value(), call->span_)
          << "WSE Calendar requires allocated L1 endpoints";
      CHECK_SPAN(tile->tile_view_.has_value(), call->span_) << "WSE Calendar requires an explicit layout";
      const auto& view = *tile->tile_view_;
      CHECK_SPAN(view.blayout == TileLayout::row_major && view.slayout == TileLayout::none_box &&
                     view.stride.empty(),
                 call->span_)
          << "WSE Calendar requires packed row-major L1 tiles";
    }
    const auto members = ParseWseMembers(call->GetKwarg<std::string>("members"));
    const auto rows = As<ConstInt>(send->shape_[0]);
    const auto cols = As<ConstInt>(send->shape_[1]);
    CHECK_SPAN(rows && cols, call->span_) << "WSE Calendar requires static final geometry";
    const int bytes = static_cast<int>((members.size() - 1) * rows->value_ * cols->value_ * 2);
    auto kwargs = call->kwargs_;
    kwargs.erase(std::remove_if(kwargs.begin(), kwargs.end(),
                                [](const auto& item) { return item.first == "exp_bytes"; }),
                 kwargs.end());
    kwargs.emplace_back("exp_bytes", bytes);
    return std::make_shared<Call>(call->op_, call->args_, kwargs, call->attrs_, call->GetType(), call->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& stmt) override {
    auto previous = result_type_;
    result_type_ = stmt->var_->GetType();
    auto result = IRMutator::VisitStmt_(stmt);
    result_type_ = previous;
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& stmt) override { return InControl(stmt); }
  StmtPtr VisitStmt_(const WhileStmtPtr& stmt) override { return InControl(stmt); }
  StmtPtr VisitStmt_(const IfStmtPtr& stmt) override { return InControl(stmt); }

 private:
  int count_ = 0;
  TypePtr result_type_;
  int control_depth_ = 0;
  template <class T>
  StmtPtr InControl(const T& stmt) {
    ++control_depth_;
    auto result = IRMutator::VisitStmt_(stmt);
    --control_depth_;
    return result;
  }
};
}  // namespace

namespace pass {
Pass FinalizeWseCalendar() {
  return CreateFunctionPass(
      [](const FunctionPtr& function) {
        FinalizeCalendar mutator;
        auto body = mutator.VisitStmt(function->body_);
        auto result = MutableCopy(function);
        result->body_ = body;
        return result;
      },
      "FinalizeWseCalendar");
}
}  // namespace pass
}  // namespace pypto::ir
