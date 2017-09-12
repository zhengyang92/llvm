//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG.  This pass is where
// algebraic simplification happens.
//
// This pass combines things like:
//    %Y = add i32 %X, 1
//    %Z = add i32 %Y, 1
// into:
//    %Z = add i32 %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. Compare instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All cmp instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//   ... etc.
//
//===----------------------------------------------------------------------===//

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

#define DEBUG_TYPE "canonicalization"


STATISTIC(NumCanonicalized , "Number of canoncalized");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
DEBUG_COUNTER(VisitCounter, "instcombine-visit",
              "Controls which instructions are visited");


bool Canonicalizer::run() {
  while (!Worklist.isEmpty()) {
    Instruction *I = Worklist.RemoveOne();
    if (I == nullptr) continue;  // skip null values.

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I, &TLI)) {
      DEBUG(dbgs() << "IC: DCE: " << *I << '\n');
      eraseInstFromFunction(*I);
      ++NumDeadInst;
      MadeIRChange = true;
      continue;
    }

    if (!DebugCounter::shouldExecute(VisitCounter))
      continue;

    // Now that we have an instruction, try combining it to simplify it.
    Builder.SetInsertPoint(I);
    Builder.SetCurrentDebugLocation(I->getDebugLoc());

#ifndef NDEBUG
    std::string OrigI;
#endif
    DEBUG(raw_string_ostream SS(OrigI); I->print(SS); OrigI = SS.str(););
    DEBUG(dbgs() << "IC: Visiting: " << OrigI << '\n');

    if (Instruction *Result = visit(*I)) {
      ++NumCanonicalized;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(dbgs() << "IC: Old = " << *I << '\n'
                     << "    New = " << *Result << '\n');

        if (I->getDebugLoc())
          Result->setDebugLoc(I->getDebugLoc());
        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Push the new instruction and any users onto the worklist.
        Worklist.AddUsersToWorkList(*Result);
        Worklist.Add(Result);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I->getIterator();

        // If we replace a PHI with something that isn't a PHI, fix up the
        // insertion point.
        if (!isa<PHINode>(Result) && isa<PHINode>(InsertPos))
          InsertPos = InstParent->getFirstInsertionPt();

        InstParent->getInstList().insert(InsertPos, Result);

        eraseInstFromFunction(*I);
      } else {
        DEBUG(dbgs() << "IC: Mod = " << OrigI << '\n'
                     << "    New = " << *I << '\n');

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I, &TLI)) {
          eraseInstFromFunction(*I);
        } else {
          Worklist.AddUsersToWorkList(*I);
          Worklist.Add(I);
        }
      }
      MadeIRChange = true;
    }
  }

  Worklist.Zap();
  return MadeIRChange;
}

/// Walk the function in depth-first order, adding all reachable code to the
/// worklist.
///
/// This has a couple of tricks to make the code faster and more powerful.  In
/// particular, we constant fold and DCE instructions as we go, to avoid adding
/// them to the worklist (this significantly speeds up instcombine on code where
/// many instructions are dead or constant).  Additionally, if we find a branch
/// whose condition is a known constant, we only visit the reachable successors.
///
static bool AddReachableCodeToWorklist(BasicBlock *BB, const DataLayout &DL,
                                       SmallPtrSetImpl<BasicBlock *> &Visited,
                                       CanonicalizationWorklist &CWorklist,
                                       const TargetLibraryInfo *TLI) {
  bool MadeIRChange = false;
  SmallVector<BasicBlock*, 256> Worklist;
  Worklist.push_back(BB);

  SmallVector<Instruction*, 128> InstrsForCanonicalizationWorklist;
  DenseMap<Constant *, Constant *> FoldedConstants;

  do {
    BB = Worklist.pop_back_val();

    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB).second)
      continue;

    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ) {
      Instruction *Inst = &*BBI++;

      // DCE instruction if trivially dead.
      if (isInstructionTriviallyDead(Inst, TLI)) {
        ++NumDeadInst;
        DEBUG(dbgs() << "IC: DCE: " << *Inst << '\n');
        Inst->eraseFromParent();
        MadeIRChange = true;
        continue;
      }

      // Skip processing debug intrinsics in InstCombine. Processing these call instructions
      // consumes non-trivial amount of time and provides no value for the optimization.
      if (!isa<DbgInfoIntrinsic>(Inst))
        InstrsForCanonicalizationWorklist.push_back(Inst);
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isConditional() && isa<ConstantInt>(BI->getCondition())) {
        bool CondVal = cast<ConstantInt>(BI->getCondition())->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        Worklist.push_back(SI->findCaseValue(Cond)->getCaseSuccessor());
        continue;
      }
    }

    for (BasicBlock *SuccBB : TI->successors())
      Worklist.push_back(SuccBB);
  } while (!Worklist.empty());

  // Once we've found all of the instructions to add to instcombine's worklist,
  // add them in reverse order.  This way instcombine will visit from the top
  // of the function down.  This jives well with the way that it adds all uses
  // of instructions to the worklist after doing a transformation, thus avoiding
  // some N^2 behavior in pathological cases.
  CWorklist.AddInitialGroup(InstrsForCanonicalizationWorklist);

  return MadeIRChange;
}

/// \brief Populate the IC worklist from a function, and prune any dead basic
/// blocks discovered in the process.
///
/// This also does basic constant propagation and other forward fixing to make
/// the combiner itself run much faster.
static bool prepareICWorklistFromFunction(Function &F, const DataLayout &DL,
                                          TargetLibraryInfo *TLI,
                                          CanonicalizationWorklist &ICWorklist) {
  bool MadeIRChange = false;

  // Do a depth-first traversal of the function, populate the worklist with
  // the reachable instructions.  Ignore blocks that are not reachable.  Keep
  // track of which blocks we visit.
  SmallPtrSet<BasicBlock *, 32> Visited;
  MadeIRChange |=
      AddReachableCodeToWorklist(&F.front(), DL, Visited, ICWorklist, TLI);

  // Do a quick scan over the function.  If we find any blocks that are
  // unreachable, remove any instructions inside of them.  This prevents
  // the instcombine code from having to deal with some bad special cases.
  for (BasicBlock &BB : F) {
    if (Visited.count(&BB))
      continue;

    unsigned NumDeadInstInBB = removeAllNonTerminatorAndEHPadInstructions(&BB);
    MadeIRChange |= NumDeadInstInBB > 0;
    NumDeadInst += NumDeadInstInBB;
  }

  return MadeIRChange;
}

static bool canonicalizationOverFunction(
    Function &F, CanonicalizationWorklist &Worklist, AliasAnalysis *AA,
    AssumptionCache &AC, TargetLibraryInfo &TLI, DominatorTree &DT,
    OptimizationRemarkEmitter &ORE,
    LoopInfo *LI = nullptr) {
  auto &DL = F.getParent()->getDataLayout();

  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<TargetFolder, IRBuilderCallbackInserter> Builder(
      F.getContext(), TargetFolder(DL),
      IRBuilderCallbackInserter([&Worklist, &AC](Instruction *I) {
        Worklist.Add(I);

        using namespace llvm::PatternMatch;
        if (match(I, m_Intrinsic<Intrinsic::assume>()))
          AC.registerAssumption(cast<CallInst>(I));
      }));

  // Lower dbg.declare intrinsics otherwise their value may be clobbered
  // by instcombiner.
  bool MadeIRChange = LowerDbgDeclare(F);

  // Iterate while there is work to do.
  int Iteration = 0;
  for (;;) {
    ++Iteration;
    DEBUG(dbgs() << "\n\nCANONICALIZATION ITERATION #" << Iteration << " on "
                 << F.getName() << "\n");

    MadeIRChange |= prepareICWorklistFromFunction(F, DL, &TLI, Worklist);

    Canonicalizer C(Worklist, Builder, F.optForMinSize(), AA,
                   AC, TLI, DT, ORE, DL, LI);

    if (C.run())
      break;
  }

  return MadeIRChange || Iteration > 1;
}

PreservedAnalyses CanonicalizationPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  auto *LI = AM.getCachedResult<LoopAnalysis>(F);

  // FIXME: The AliasAnalysis is not yet supported in the new pass manager
  if (!canonicalizationOverFunction(F, Worklist, nullptr, AC, TLI, DT, ORE, LI))
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();

  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

void CanonicalizationLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  }

bool CanonicalizationLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  // Required analyses.
  auto AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // Optional analyses.
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;

  return canonicalizationOverFunction
    (F, Worklist, AA, AC, TLI, DT, ORE,LI);
}

char CanonicalizationLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CanonicalizationLegacyPass, "canonicalization",
                      "Canonicalize instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(CanonicalizationLegacyPass, "canonicalization",
                    "Canonicalize instructions", false, false)

// Initialization Routines
void llvm::initializeCanonicalizationLegacy(PassRegistry &Registry) {
  initializeCanonicalizationLegacyPassPass(Registry);
}

void LLVMInitializeCanonicalizationLegacy(LLVMPassRegistryRef R) {
  initializeCanonicalizationLegacyPassPass(*unwrap(R));
}

FunctionPass *
llvm::createCanonicalizationLegacyPass() {
  return new CanonicalizationLegacyPass();
}
