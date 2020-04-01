//===- Pass.cpp - MLIR pass registration generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PassGen uses the description of passes to generate base classes for passes
// and command line registration.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// GEN: Pass registration generation
//===----------------------------------------------------------------------===//

/// Emit the code for registering each of the given passes with the global
/// PassRegistry.
static void emitRegistration(ArrayRef<Pass> passes, raw_ostream &os) {
  os << "#ifdef GEN_PASS_REGISTRATION\n";
  for (const Pass &pass : passes) {
    os << llvm::formatv("::mlir::registerPass(\"{0}\", \"{1}\", []() -> "
                        "std::unique_ptr<Pass> {{ return {2}; });\n",
                        pass.getArgument(), pass.getSummary(),
                        pass.getConstructor());
  }
  os << "#undef GEN_PASS_REGISTRATION\n";
  os << "#endif // GEN_PASS_REGISTRATION\n";
}

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

static void emitDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit */\n";

  std::vector<Pass> passes;
  for (const llvm::Record *def : recordKeeper.getAllDerivedDefinitions("Pass"))
    passes.push_back(Pass(def));
  emitRegistration(passes, os);
}

static mlir::GenRegistration
    genRegister("gen-pass-decls", "Generate operation documentation",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  emitDecls(records, os);
                  return false;
                });
