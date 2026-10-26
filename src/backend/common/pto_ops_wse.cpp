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

#include <sstream>
#include <string>
#include <unordered_set>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/wse.h"
#include "src/backend/common/pto_ops_internal.h"

namespace pypto::backend {
void RegisterWseOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  if (exclude_ops.count("tile.wse_allgather")) return;
  backend.RegisterOp("tile.wse_allgather")
      .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& base) -> std::string {
        auto& cg = pto_ops_detail::AsPto(base);
        auto src = op->args_[0];
        auto input = ir::As<ir::TileType>(src->GetType());
        auto members = ir::ParseWseMembers(op->GetKwarg<std::string>("members"));
        int axis = op->GetKwarg<int>("axis");
        int64_t expected = (members.size() - 1) * ir::As<ir::ConstInt>(input->shape_[0])->value_ *
                           ir::As<ir::ConstInt>(input->shape_[1])->value_ * input->dtype_.GetByte();
        INTERNAL_CHECK_SPAN(op->GetKwarg<int>("exp_bytes") == expected, op->span_)
            << "WSE receive contract changed after lowering";
        auto exp = cg.GetOrEmitConstant(expected, DataType::INT32);
        auto dst = cg.GetCurrentResultTarget();
        std::ostringstream attrs;
        attrs << "{axis = " << axis << " : i64, members = array<i64: ";
        for (size_t i = 0; i < members.size(); ++i) {
          if (i) attrs << ", ";
          attrs << members[i];
        }
        attrs << ">}";
        cg.Emit("\"pto.tcalendar\"(%__wse_context, " + cg.GetExprAsCode(src) + ", " + dst + ", " + exp +
                ") " + attrs.str() + " : (i64, " + cg.GetExprTypeAnnotation(src) + ", " +
                cg.GetCurrentResultTileBufTypeString() + ", i32) -> ()");
        cg.SetCurrentExprValue(dst);
        return "";
      });
}
}  // namespace pypto::backend
