//
//  Created by Toru Tamaki on 2013/11/28.
//  Copyright (c) 2013 tamaki. All rights reserved.
//

#ifndef opengmtest_infer_hxx
#define opengmtest_infer_hxx


if (use_BP) {

    typedef opengm::BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
    typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
    const size_t maxNumberOfIterations = 40;
    const double convergenceBound = 1e-7;
    const double damping = 0.5;
    BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
    BeliefPropagation bp(gm, parameter);
    BeliefPropagation::VerboseVisitorType visitor;
    bp.infer(visitor);
    bp.arg(labeling);
    showMarginals(labeling, bp, nx, sigma);
}

#ifdef WITH_MAXFLOW
else if (use_ab_graphcut) {
    typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaBetaSwap<Model, MinGraphCut> MinAlphaBetaSwap;
    MinAlphaBetaSwap abs(gm);
    MinAlphaBetaSwap::VerboseVisitorType visitor;
    abs.infer(visitor);
    abs.arg(labeling);
}

else if (use_aexp_graphcut) {
    typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;
    MinAlphaExpansion aexp(gm);
    MinAlphaExpansion::VerboseVisitorType visitor;
    aexp.infer(visitor);
    aexp.arg(labeling);
}
#endif

#ifdef WITH_MAXFLOW_IBFS
else if (use_ab_graphcut_ibfs) {
    typedef opengm::external::MinSTCutIBFS<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaBetaSwap<Model, MinGraphCut> MinAlphaBetaSwap;
    MinAlphaBetaSwap abs(gm);
    MinAlphaBetaSwap::VerboseVisitorType visitor;
    abs.infer(visitor);
    abs.arg(labeling);
}

else if (use_aexp_graphcut_ibfs) {
    typedef opengm::external::MinSTCutIBFS<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;
    MinAlphaExpansion aexp(gm);
    MinAlphaExpansion::VerboseVisitorType visitor;
    aexp.infer(visitor);
    aexp.arg(labeling);
}
#endif

#ifdef WITH_BOOST
else if (use_ab_graphcut_boost) {
    typedef opengm::MinSTCutBoost<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaBetaSwap<Model, MinGraphCut> MinAlphaBetaSwap;
    MinAlphaBetaSwap abs(gm);
    MinAlphaBetaSwap::VerboseVisitorType visitor;
    abs.infer(visitor);
    abs.arg(labeling);
}

else if (use_aexp_graphcut_boost) {
    typedef opengm::MinSTCutBoost<size_t, double> MinStCutType;
    typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
    typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;
    MinAlphaExpansion aexp(gm);
    MinAlphaExpansion::VerboseVisitorType visitor;
    aexp.infer(visitor);
    aexp.arg(labeling);
}
#endif

else if (use_icm) {
    typedef opengm::ICM<Model, opengm::Minimizer> MinICM;
    MinICM icm(gm);
    MinICM::VerboseVisitorType visitor;
    icm.infer(visitor);
    icm.arg(labeling);
}

else if (use_lazyFlipper) {
    typedef opengm::LazyFlipper<Model, opengm::Minimizer> LazyFlipper;
    LazyFlipper lf(gm);
    LazyFlipper::VerboseVisitorType visitor;
    lf.infer(visitor);
    lf.arg(labeling);
}

#ifdef WITH_AD3
else if (use_loc) {
    typedef opengm::LOC<Model, opengm::Minimizer> LOC;
    LOC::Parameter parameter;
    parameter.phi_ = 5;
    //parameter.maxRadius_ = 10; // obsolated?
    parameter.maxIterations_ = 100;
    LOC loc(gm, parameter);
    LOC::VerboseVisitorType visitor;
    loc.infer(visitor);
    loc.arg(labeling);
}
#endif

else if (use_TRBP) {
    typedef opengm::TrbpUpdateRules<Model, opengm::Minimizer> UpdateRules;
    typedef opengm::MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;
    const size_t maxNumberOfIterations = 100;
    const double convergenceBound = 1e-7;
    const double damping = 0.0;
    TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
    TRBP trbp(gm, parameter);
    TRBP::VerboseVisitorType visitor;
    trbp.infer(visitor);
    trbp.arg(labeling);
    showMarginals(labeling, trbp, nx, sigma);
}

#ifdef WITH_TRWS
else if (use_TRWS) {
    typedef opengm::TrbpUpdateRules<Model, opengm::Minimizer> UpdateRules;
    typedef opengm::external::TRWS<Model> TRWS;
    TRWS::Parameter parameter;
    parameter.tolerance_ = 1e-7;
    TRWS trws(gm, parameter);
    TRWS::VerboseVisitorType visitor;
    trws.infer(visitor);
    trws.arg(labeling);
}
#endif

#ifdef WITH_SAMPLING
else if (use_Gibbs) {
    typedef opengm::Gibbs<Model, opengm::Minimizer> Gibbs;
    const size_t numberOfSamplingSteps = 1e4;
    const size_t numberOfBurnInSteps = 1e4;
    Gibbs::Parameter parameter(numberOfSamplingSteps, numberOfBurnInSteps);
    parameter.startPoint_ = labeling;
    Gibbs gibbs(gm, parameter);
    Gibbs::VerboseVisitorType visitor;
    gibbs.infer(visitor);
    gibbs.arg(labeling);
}

else if (use_SwendsenWang) {
    typedef opengm::SwendsenWang<Model, opengm::Minimizer> SwendsenWang;
    SwendsenWang::Parameter parameter;
    parameter.maxNumberOfSamplingSteps_ = 1e4;
    parameter.numberOfBurnInSteps_ = 1e4;
    parameter.initialState_ = labeling;
    SwendsenWang sw(gm, parameter);
    SwendsenWang::VerboseVisitorType visitor;
    sw.infer(visitor);
    sw.arg(labeling);
}
#endif

#ifdef WITH_DP
else if (use_DP) {
    typedef opengm::DynamicProgramming<Model, opengm::Minimizer> DP;
    DP dp(gm);
    DP::VerboseVisitorType visitor;
    dp.infer(visitor);
    dp.arg(labeling);
}
#endif

#ifdef WITH_QPBO
else if (use_MQPBO) {
    typedef opengm::MQPBO<Model, opengm::Minimizer> MQPBO; // space must be DiscreteSpace, not SimpleDiscreteSpace
    MQPBO mqpbo(gm);
    MQPBO::VerboseVisitorType visitor;
    mqpbo.infer(visitor);
    mqpbo.arg(labeling);
}

else if (use_aexp_fusion) {
    typedef opengm::AlphaExpansionFusion<Model, opengm::Minimizer> AlphaExpFusion;
    AlphaExpFusion aexp_fusion(gm);
    AlphaExpFusion::VerboseVisitorType visitor;
    aexp_fusion.infer(visitor);
    aexp_fusion.arg(labeling);
}

//    //===================================================== can't compile
//    typedef opengm::external::QPBO<Model> MinQPBO;
//    typedef opengm::AlphaBetaSwap<Model, MinQPBO> MinAlphaBetaSwap;
//
//    MinAlphaBetaSwap abs(gm);
//    MinAlphaBetaSwap::VerboseVisitorType visitor;
//    abs.infer(visitor);
//
//    // obtain the (approximate) argmin
//    abs.arg(labeling);
//    //=====================================================

#endif

#ifdef WITH_MRF
else if (use_MRFLIB_ICM) {
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::ICM;
    para.energyType_ = MRFLIB::Parameter::WEIGHTEDTABLE;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}

else if (use_MRFLIB_aexp) { // segmentation fault
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::EXPANSION;
    para.energyType_ = MRFLIB::Parameter::VIEW;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}

else if (use_MRFLIB_ab) { // segmentation fault
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::SWAP;
    para.energyType_ = MRFLIB::Parameter::VIEW;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}

else if (use_MRFLIB_LBP) {
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::MAXPRODBP;
    para.energyType_ = MRFLIB::Parameter::TABLES;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}

else if (use_MRFLIB_BPS) {
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::BPS;
    para.energyType_ = MRFLIB::Parameter::VIEW;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}

else if (use_MRFLIB_TRWS) {
    typedef opengm::external::MRFLIB<Model> MRFLIB;
    MRFLIB::Parameter para;
    para.inferenceType_ = MRFLIB::Parameter::TRWS;
    para.energyType_ = MRFLIB::Parameter::VIEW;
    para.numberOfIterations_ = 10;
    MRFLIB mrf(gm,para);
    MRFLIB::VerboseVisitorType visitor;
    mrf.infer(visitor);
    mrf.arg(labeling);
}
#endif

else if (use_DD_subgradient) {
    typedef opengm::DDDualVariableBlock<marray::Marray<double> > DualBlockType;
    typedef opengm::DualDecompositionBase<Model, DualBlockType>::SubGmType SubGmType;
    typedef opengm::BeliefPropagationUpdateRules<SubGmType, opengm::Minimizer> UpdateRuleType;
    typedef opengm::MessagePassing<SubGmType, opengm::Minimizer, UpdateRuleType, opengm::MaxDistance> InfType;
    typedef opengm::DualDecompositionSubGradient<Model, InfType, DualBlockType> DualDecompositionSubGradient;
    DualDecompositionSubGradient::Parameter param;
    param.useProjectedAdaptiveStepsize_ = TRUE;
    DualDecompositionSubGradient ddsg(gm, param);
    DualDecompositionSubGradient::VerboseVisitorType visitor;
    ddsg.infer(visitor);
    ddsg.arg(labeling);
}

#ifdef WITH_LIBDAI
else if (use_libdai_BP) {
    typedef opengm::external::libdai::Bp<Model, opengm::Minimizer> BP;
    const size_t maxIterations=100;
    const double damping=0.0;
    const double tolerance=0.000001;
    BP::UpdateRule updateRule = BP::PARALL; // Bp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX 9
    size_t verboseLevel=10;
    BP::Parameter parameter(maxIterations, damping, tolerance, updateRule, verboseLevel);
    BP bp(gm, parameter);
    BP::VerboseVisitorType visitor;
    bp.infer(visitor);
    bp.arg(labeling);
}

else if (use_libdai_TRBP) {
    typedef opengm::external::libdai::TreeReweightedBp<Model, opengm::Minimizer> Trbp;
    const size_t maxIterations=100;
    const double damping=0.0;
    const size_t ntrees=0;
    const double tolerance=0.000001;
    Trbp::UpdateRule updateRule = Trbp::PARALL; // Trbp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX 9
    size_t verboseLevel=10;
    Trbp::Parameter parameter(maxIterations, damping, tolerance, ntrees, updateRule, verboseLevel);
    Trbp trbp(gm, parameter);
    Trbp::VerboseVisitorType visitor;
    trbp.infer(visitor);
    trbp.arg(labeling);
}

else if (use_libdai_DLGBP) {
    typedef opengm::external::libdai::DoubleLoopGeneralizedBP<Model, opengm::Minimizer> DoubleLoopGeneralizedBP;
    const bool doubleloop=1;
    // DoubleLoopGeneralizedBP::Clusters=MIN | BETHE | DELTA | LOOP
    const DoubleLoopGeneralizedBP::Clusters clusters = DoubleLoopGeneralizedBP::BETHE;
    const size_t loopdepth = 3;
    // DoubleLoopGeneralizedBP::Init = UNIFORM | RANDOM
    const DoubleLoopGeneralizedBP::Init init = DoubleLoopGeneralizedBP::UNIFORM;
    const size_t maxiter=3;//10000;
    const double tolerance=1e-9;
    const size_t verboseLevel=10;
    DoubleLoopGeneralizedBP::Parameter parameter(doubleloop, clusters, loopdepth, init, maxiter, tolerance, verboseLevel);
    DoubleLoopGeneralizedBP gdlbp(gm, parameter);
    DoubleLoopGeneralizedBP::VerboseVisitorType visitor;
    gdlbp.infer(visitor);
    gdlbp.arg(labeling);
}

else if (use_libdai_FBP) {
    typedef opengm::external::libdai::FractionalBp<Model, opengm::Minimizer> FractionalBp;
    const size_t maxIterations=100;
    const double damping=0.0;
    const double tolerance=0.000001;
    // FractionalBp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX
    FractionalBp::UpdateRule updateRule = FractionalBp::PARALL;
    size_t verboseLevel=10;
    FractionalBp::Parameter parameter(maxIterations, damping, tolerance, updateRule, verboseLevel);
    FractionalBp fbp(gm, parameter);
    FractionalBp::VerboseVisitorType visitor;
    fbp.infer(visitor);
    fbp.arg(labeling);
}

else if (use_libdai_JT) {
    typedef opengm::external::libdai::JunctionTree<Model, opengm::Minimizer> JunctionTree;
    // JunctionTree::UpdateRule = HUGIN | SHSH
    JunctionTree::UpdateRule updateRule = JunctionTree::HUGIN;
    // JunctionTree::Heuristic = MINFILL | WEIGHTEDMINFILL | MINWEIGHT | MINNEIGHBORS
    JunctionTree::Heuristic heuristic = JunctionTree::MINWEIGHT;
    size_t verboseLevel=10;
    JunctionTree::Parameter parameter(updateRule, heuristic,verboseLevel);
    JunctionTree jt(gm, parameter);
    JunctionTree::VerboseVisitorType visitor;
    jt.infer(visitor);
    jt.arg(labeling);
}

else if (use_libdai_TEP) {
    typedef opengm::external::libdai::TreeExpectationPropagation<Model, opengm::Minimizer> TreeExpectationPropagation;
    //TreeExpectationPropagation::TreeEpType = ORG | ALT
    TreeExpectationPropagation::TreeEpType treeEpTyp=TreeExpectationPropagation::ORG;
    const size_t maxiter=100;
    const double tolerance=1e-9;
    size_t verboseLevel=10;
    TreeExpectationPropagation::Parameter parameter(treeEpTyp, maxiter, tolerance, verboseLevel);
    TreeExpectationPropagation treeep(gm, parameter);
    TreeExpectationPropagation::VerboseVisitorType visitor;
    treeep.infer(visitor);
    treeep.arg(labeling);
}

else if (use_libdai_Gibbs) {
    typedef opengm::external::libdai::Gibbs<Model, opengm::Minimizer> Gibbs;
    const size_t maxiter = 100;
    const size_t burnin  = 10;
    const size_t restart = 20;
    const size_t verbose = 0;
    Gibbs::Parameter parameter(maxiter, burnin, restart, verbose);
    Gibbs gibbs(gm, parameter);
    Gibbs::VerboseVisitorType visitor;
    gibbs.infer(visitor);
    gibbs.arg(labeling);
}

else if (use_libdai_MF) {  // fixed; libc++abi.dylib: terminate called throwing an exception
    typedef opengm::external::libdai::MeanField<Model, opengm::Minimizer> MeanField;
    const size_t maxiter=10000;
    const double damping=0.2;
    const double tolerance=1e-1;
    // MeanField::UpdateRule = NAIVE | HARDSPIN
    const MeanField::UpdateRule updateRule = MeanField::NAIVE;
    // MeanField::Init = UNIFORM | RANDOM
    const MeanField::Init init = MeanField::UNIFORM;
    const size_t verboseLevel=10;
    MeanField::Parameter parameter(maxiter,damping,tolerance,updateRule,init,verboseLevel);
    MeanField mf(gm, parameter);
    MeanField::EmptyVisitorType visitor;
    mf.infer(visitor);
    //mf.arg(labeling); // does not work
    showMarginals(labeling, mf, nx, sigma);
}

#endif

#ifdef WITH_FASTPD
else if (use_fastPD) {
    typedef opengm::external::FastPD<Model> fastPD;
    fastPD::Parameter param;
    param.numberOfIterations_ = 1000;
    fastPD fastpd(gm, param);
    fastPD::VerboseVisitorType visitor;
    fastpd.infer(visitor);
    fastpd.arg(labeling);
}
#endif

#ifdef WITH_MPLP
else if (use_MPLP) {
    typedef opengm::external::MPLP<Model> Mplp;
    Mplp::Parameter param;
    param.maxTime_ = 120;
    Mplp mplp(gm, param);
    Mplp::VerboseVisitorType visitor;
    mplp.infer(visitor);
    mplp.arg(labeling);
}
#endif

#ifdef WITH_GCO
else if (use_GCO_aexp) {
    typedef opengm::external::GCOLIB<Model> GCO;
    GCO::Parameter param;
    param.inferenceType_ = GCO::Parameter::EXPANSION; // EXPANSION, SWAP
    param.energyType_ = GCO::Parameter::VIEW; // VIEW, TABLES, WEIGHTEDTABLE
    param.useAdaptiveCycles_ = false; // for alpha-expansion
    GCO gco(gm, param);
    GCO::VerboseVisitorType visitor;
    gco.infer(visitor);
    gco.arg(labeling);
}

else if (use_GCO_swap) {
    typedef opengm::external::GCOLIB<Model> GCO;
    GCO::Parameter param;
    param.inferenceType_ = GCO::Parameter::SWAP; // EXPANSION, SWAP
    param.energyType_ = GCO::Parameter::VIEW; // VIEW, TABLES, WEIGHTEDTABLE
    GCO gco(gm, param);
    GCO::VerboseVisitorType visitor;
    gco.infer(visitor);
    gco.arg(labeling);
}
#endif


#if defined(WITH_QPBO) && defined(WITH_AD3)
//int use_fusion_move = 0;
else if (0) { // still unknown how to use
    typedef opengm::FusionMover<Model, opengm::Minimizer> FusionMover;
    FusionMover fusionmove(gm);
}
#endif












#endif
