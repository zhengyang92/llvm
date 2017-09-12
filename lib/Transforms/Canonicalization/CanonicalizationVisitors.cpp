#include "CanonicalizationInternal.h"
#include "llvm-c/Initialization.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Canonicalization/Canonicalization.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <climits>
using namespace llvm;
using namespace llvm::PatternMatch;

Instruction *Canonicalizer::visitBinaryOperator(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  auto *Op0Const = dyn_cast<Constant>(Op0);
  auto *Op1Const = dyn_cast<Constant>(Op1);
  if (Op0Const && !Op1Const) {
    std::swap(Op0, Op1);
  }
  return nullptr;
}

Instruction *Canonicalizer::visitMul(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // X * -1 == 0 - X
  if (match(Op1, m_AllOnes())) {
    BinaryOperator *BO = BinaryOperator::CreateNeg(Op0, I.getName());
    if (I.hasNoSignedWrap())
      BO->setHasNoSignedWrap();
    return BO;
  }

  if (isa<Constant>(Op1)) {
    //    if (Instruction *FoldedMul = foldOpWithConstantIntoOperand(I))
    //      return FoldedMul;

    // Canonicalize (X+C1)*CI -> X*CI+C1*CI.
    {
      Value *X;
      Constant *C1;
      if (match(Op0, m_OneUse(m_Add(m_Value(X), m_Constant(C1))))) {
        Value *Mul = Builder.CreateMul(C1, Op1);
        // Only go forward with the transform if C1*CI simplifies to a tidier
        // constant.
        if (!match(Mul, m_Mul(m_Value(), m_Value())))
          return BinaryOperator::CreateAdd(Builder.CreateMul(X, Op1), Mul);
      }
    }
  }

  /// i1 mul -> i1 and. //??
  /*  if (I.getType()->isIntOrIntVectorTy(1))
      return BinaryOperator::CreateAnd(Op0, Op1);*/

  return nullptr;
}




/// \brief Creates node of binary operation with the same attributes as the
/// specified one but with other operands.

/// \brief Makes transformation of binary operation specific for vector types.
/// \param Inst Binary operator to transform.
/// \return Pointer to node that must replace the original binary operator, or
///         null pointer if no transformation was made.

Instruction *Canonicalizer::visitGetElementPtrInst(GetElementPtrInst &GEP) {

  Value *PtrOp = GEP.getOperand(0);

  if (GEP.getNumIndices() == 1) {
    unsigned AS = GEP.getPointerAddressSpace();
    if (GEP.getOperand(1)->getType()->getScalarSizeInBits() ==
        DL.getPointerSizeInBits(AS)) {
      Type *Ty = GEP.getSourceElementType();
      uint64_t TyAllocSize = DL.getTypeAllocSize(Ty);

      bool Matched = false;
      uint64_t C;
      Value *V = nullptr;
      if (TyAllocSize == 1) {
        V = GEP.getOperand(1);
        Matched = true;
      } else if (match(GEP.getOperand(1),
                       m_AShr(m_Value(V), m_ConstantInt(C)))) {
        if (TyAllocSize == 1ULL << C)
          Matched = true;
      } else if (match(GEP.getOperand(1),
                       m_SDiv(m_Value(V), m_ConstantInt(C)))) {
        if (TyAllocSize == C)
          Matched = true;
      }

      if (Matched) {
        // Canonicalize (gep i8* X, -(ptrtoint Y))
        // to (inttoptr (sub (ptrtoint X), (ptrtoint Y)))
        // The GEP pattern is emitted by the SCEV expander for certain kinds of
        // pointer arithmetic.
        if (match(V, m_Neg(m_PtrToInt(m_Value())))) {
          Operator *Index = cast<Operator>(V);
          Value *PtrToInt = Builder.CreatePtrToInt(PtrOp, Index->getType());
          Value *NewSub = Builder.CreateSub(PtrToInt, Index->getOperand(1));
          return CastInst::Create(Instruction::IntToPtr, NewSub, GEP.getType());
        }
        // Canonicalize (gep i8* X, (ptrtoint Y)-(ptrtoint X))
        // to (bitcast Y)
        Value *Y;
        if (match(V, m_Sub(m_PtrToInt(m_Value(Y)),
                           m_PtrToInt(m_Specific(GEP.getOperand(0)))))) {
          return CastInst::CreatePointerBitCastOrAddrSpaceCast(Y,
                                                               GEP.getType());
        }
      }
    }
  }

  return nullptr;
}


Instruction *Canonicalizer::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  BasicBlock *TrueDest;
  BasicBlock *FalseDest;

  // Canonicalize, for example, icmp_ne -> icmp_eq or fcmp_one -> fcmp_oeq.
  CmpInst::Predicate Pred;
  if (match(&BI, m_Br(m_OneUse(m_Cmp(Pred, m_Value(), m_Value())), TrueDest,
                      FalseDest)) &&
      !isCanonicalPredicate(Pred)) {
    // Swap destinations and condition.
    CmpInst *Cond = cast<CmpInst>(BI.getCondition());
    Cond->setPredicate(CmpInst::getInversePredicate(Pred));
    BI.swapSuccessors();
    Worklist.Add(Cond);
    return &BI;
  }

  return nullptr;
}

