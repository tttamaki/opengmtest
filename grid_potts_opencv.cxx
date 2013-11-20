//
// taken from
// opengm/src/examples/image-processing-examples/grid_potts.cxx
//
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/swendsenwang.hxx>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>

#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <opengm/inference/loc.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/mqpbo.hxx>


#include <opengm/inference/external/qpbo.hxx>
#include <opengm/inference/external/trws.hxx>



#include <opengm/inference/external/mrflib.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>

#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_expectation_propagation.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/gibbs.hxx>
#include <opengm/inference/external/libdai/mean_field.hxx>



void
imshow(const std::string &title,
       const std::vector<Eigen::MatrixXf> gridvec,
       const float maxval)
{
    
    cv::Mat _rgb[3], rgb;
    cv::eigen2cv(gridvec[0], _rgb[0]);
    cv::eigen2cv(gridvec[1], _rgb[1]);
    cv::eigen2cv(gridvec[2], _rgb[2]);
    
    
    cv::merge(_rgb, 3, rgb);
    cv::imshow(title, rgb);
    
}





//using namespace std; // 'using' is used only in example code
using namespace opengm;

// model parameters (global variables are used only in example code)
const size_t nx = 30; // width of the grid
const size_t ny = 30; // height of the grid
const size_t numberOfLabels = 3; // just for RGB
double lambda = 0.1; // coupling strength of the Potts model

// this function maps a node (x, y) in the grid to a unique variable index
inline size_t variableIndex(const size_t x, const size_t y) {
    return x + nx * y;
}

int main() {
    
    
    
    
    cv::namedWindow("control panel", CV_WINDOW_NORMAL | CV_GUI_EXPANDED);

    int use_BP = 0;
    cv::createTrackbar("BP", "control panel", &use_BP, 1,  NULL);
    int use_ab_graphcut = 1;
    cv::createTrackbar("a-b GC", "control panel", &use_ab_graphcut, 1,  NULL);
    int use_aexp_graphcut = 0;
    cv::createTrackbar("a-exp GC", "control panel", &use_aexp_graphcut, 1,  NULL);
    int use_icm = 0;
    cv::createTrackbar("ICM", "control panel", &use_icm, 1,  NULL);
    int use_lazyFlipper = 0;
    cv::createTrackbar("LazyFlipper", "control panel", &use_lazyFlipper, 1,  NULL);
    int use_loc = 0;
    cv::createTrackbar("LOC", "control panel", &use_loc, 1,  NULL);

    int use_TRBP = 0;
    cv::createTrackbar("TRBP", "control panel", &use_TRBP, 1,  NULL);
    int use_TRWS = 0;
    cv::createTrackbar("TRWS", "control panel", &use_TRWS, 1,  NULL);
    int use_Gibbs = 0;
    cv::createTrackbar("Gibbs", "control panel", &use_Gibbs, 1,  NULL);
    int use_SwendsenWang = 0;
    cv::createTrackbar("SW", "control panel", &use_SwendsenWang, 1,  NULL);
    
    int use_DP = 0;
    cv::createTrackbar("DP", "control panel", &use_DP, 1,  NULL);
    int use_MQPBO = 0;
    cv::createTrackbar("MQPBO", "control panel", &use_MQPBO, 1,  NULL);
    
    int use_MRFLIB_ICM = 0;
    cv::createTrackbar("MRFLIB ICM", "control panel", &use_MRFLIB_ICM, 1,  NULL);
    int use_MRFLIB_aexp = 0;
    cv::createTrackbar("MRFLIB a-exp", "control panel", &use_MRFLIB_aexp, 1,  NULL);
    int use_MRFLIB_ab = 0;
    cv::createTrackbar("MRFLIB ab", "control panel", &use_MRFLIB_ab, 1,  NULL);
    int use_MRFLIB_LBP = 0;
    cv::createTrackbar("MRFLIB LBP", "control panel", &use_MRFLIB_LBP, 1,  NULL);
    int use_MRFLIB_BPS = 0;
    cv::createTrackbar("MRFLIB BPS", "control panel", &use_MRFLIB_BPS, 1,  NULL);
    int use_MRFLIB_TRWS = 0;
    cv::createTrackbar("MRFLIB TRWS", "control panel", &use_MRFLIB_TRWS, 1,  NULL);

    int use_DD_subgradient = 0;
    cv::createTrackbar("DD subgrad", "control panel", &use_DD_subgradient, 1,  NULL);

    int use_libdai_BP = 0;
    cv::createTrackbar("libdai BP", "control panel", &use_libdai_BP, 1,  NULL);
    int use_libdai_TRBP = 0;
    cv::createTrackbar("libdai TRBP", "control panel", &use_libdai_TRBP, 1,  NULL);
    int use_libdai_DLGBP = 0;
    cv::createTrackbar("libdai DLGBP", "control panel", &use_libdai_DLGBP, 1,  NULL);
    int use_libdai_FBP = 0;
    cv::createTrackbar("libdai FBP", "control panel", &use_libdai_FBP, 1,  NULL);
    int use_libdai_JT = 0;
    cv::createTrackbar("libdai JT", "control panel", &use_libdai_JT, 1,  NULL);
    int use_libdai_TEP = 0;
    cv::createTrackbar("libdai TEP", "control panel", &use_libdai_TEP, 1,  NULL);
    int use_libdai_Gibbs = 0;
    cv::createTrackbar("libdai Gibbs", "control panel", &use_libdai_Gibbs, 1,  NULL);
    int use_libdai_MF = 0;
    cv::createTrackbar("libdai MF", "control panel", &use_libdai_MF, 1,  NULL);

    
    
    
    
    
    
    
    
    
    int noise = 2;
    cv::createTrackbar("noise*10", "control panel", &noise, 10,  NULL);
    int recreate = 1;
    cv::createTrackbar("re-create", "control panel", &recreate, 1,  NULL);

    
    
    
    std::vector<Eigen::MatrixXf> gridvec;
    
    
    
    while(1){
        

        
        
        // construct a label space with
        // - nx * ny variables
        // - each having numberOfLabels many labels
//        typedef SimpleDiscreteSpace<size_t, size_t> Space;  // mqpbo does not accept this.
        typedef DiscreteSpace<size_t, size_t> Space;
        Space space(nx * ny, numberOfLabels);
        
        if(recreate == 1){
            gridvec.clear();
            double c[] = {ny/4, ny/2, ny/4*3};
            for(size_t s = 0; s < numberOfLabels; ++s) {
                Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,ny);
                for(size_t y = 0; y < ny; ++y)
                    for(size_t x = 0; x < nx; ++x) {
                        mygrid(y,x) = 0.8 * exp( - ((c[s] - x)*(c[s] - x) + (c[s] - y)*(c[s] - y)) / (ny*2) );
                        mygrid(y,x) += noise/10. * (rand() / (double)RAND_MAX);
                    }
                gridvec.push_back(mygrid);
            }

            cv::setTrackbarPos("re-create", "control panel", 0);
        }
        imshow("RGB", gridvec, 1.0);
        
        
        // construct a graphical model with
        // - addition as the operation (template parameter Adder)
        // - support for Potts functions (template parameter PottsFunction<double>)
        typedef GraphicalModel<double, Adder,
        OPENGM_TYPELIST_2(ExplicitFunction<double> , PottsFunction<double> ) ,
        Space> Model;
        Model gm(space);
        
        // for each node (x, y) in the grid, i.e. for each variable
        // variableIndex(x, y) of the model, add one 1st order functions
        // and one 1st order factor
        for(size_t y = 0; y < ny; ++y)
            for(size_t x = 0; x < nx; ++x) {
                // function
                const size_t shape[] = {numberOfLabels};
                ExplicitFunction<double> f(shape, shape + 1);
                for(size_t s = 0; s < numberOfLabels; ++s) {
                    double tmp = 1.0 - gridvec[s](y,x); //rand() / (double)RAND_MAX;
                    f(s) = (1.0 - lambda) * tmp;
                }
                Model::FunctionIdentifier fid = gm.addFunction(f);
                
                // factor
                size_t variableIndices[] = {variableIndex(x, y)};
                gm.addFactor(fid, variableIndices, variableIndices + 1);
            }
        
        // add one (!) 2nd order Potts function
        const double valEqual = 0.0;
        const double valUnequal = lambda;
        PottsFunction<double> f(numberOfLabels, numberOfLabels, valEqual, valUnequal);
        Model::FunctionIdentifier fid = gm.addFunction(f);
        
        // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
        // add one factor that connects the corresponding variable indices and
        // refers to the Potts function
        for(size_t y = 0; y < ny; ++y)
            for(size_t x = 0; x < nx; ++x) {
                if(x + 1 < nx) { // (x, y) -- (x + 1, y)
                    size_t variableIndices[] = {variableIndex(x, y), variableIndex(x + 1, y)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
                if(y + 1 < ny) { // (x, y) -- (x, y + 1)
                    size_t variableIndices[] = {variableIndex(x, y), variableIndex(x, y + 1)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
            }

        

        std::vector<size_t> labeling(nx * ny);
        
        if (use_BP) {
            //=====================================================
            // set up the optimizer (loopy belief propagation)
            typedef BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
            typedef MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
            const size_t maxNumberOfIterations = 40;
            const double convergenceBound = 1e-7;
            const double damping = 0.5;
            BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
            BeliefPropagation bp(gm, parameter);
            
            // optimize (approximately)
            BeliefPropagation::VerboseVisitorType visitor;
            bp.infer(visitor);
            
            // obtain the (approximate) argmin
            
            bp.arg(labeling);
            //=====================================================
        }
        
        else if (use_ab_graphcut) {
            //=====================================================
            typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
            typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
            typedef opengm::AlphaBetaSwap<Model, MinGraphCut> MinAlphaBetaSwap;
            
            MinAlphaBetaSwap abs(gm);
            MinAlphaBetaSwap::VerboseVisitorType visitor;
            abs.infer(visitor);
            
            // obtain the (approximate) argmin
            abs.arg(labeling);
            //=====================================================
        }
        
        else if (use_aexp_graphcut) {
            //=====================================================
            typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
            typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
            typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;
            
            MinAlphaExpansion aexp(gm);
            MinAlphaExpansion::VerboseVisitorType visitor;
            aexp.infer(visitor);
            
            // obtain the (approximate) argmin
            aexp.arg(labeling);
            //=====================================================
        }
        
        else if (use_icm) {
            //=====================================================
            typedef opengm::ICM<Model, opengm::Minimizer> MinICM;
            
            MinICM icm(gm);
            MinICM::VerboseVisitorType visitor;
            icm.infer(visitor);
            
            // obtain the (approximate) argmin
            icm.arg(labeling);
            //=====================================================
        }
        
        else if (use_lazyFlipper) {
            //=====================================================
            typedef opengm::LazyFlipper<Model, opengm::Minimizer> LazyFlipper;
            
            LazyFlipper lf(gm);
            LazyFlipper::VerboseVisitorType visitor;
            lf.infer(visitor);
            
            // obtain the (approximate) argmin
            lf.arg(labeling);
            //=====================================================
        }
        
        
        else if (use_loc) {
            //=====================================================
            typedef opengm::LOC<Model, opengm::Minimizer> LOC;
            LOC::Parameter parameter;
            parameter.phi_ = 5;
            parameter.maxRadius_ = 10;
            parameter.maxIterations_ = 100;
            
            LOC loc(gm, parameter);
            LOC::VerboseVisitorType visitor;
            loc.infer(visitor);
            
            // obtain the (approximate) argmin
            loc.arg(labeling);
            //=====================================================
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

        
        else if (use_TRBP) {
            //=====================================================
            // set up the optimizer (tree re−weighted belief propagation)
            typedef TrbpUpdateRules<Model, opengm::Minimizer> UpdateRules;
            typedef MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;
            const size_t maxNumberOfIterations = 100;
            const double convergenceBound = 1e-7;
            const double damping = 0.0;
            TRBP::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
            TRBP trbp(gm, parameter);
            
            TRBP::VerboseVisitorType visitor;
            trbp.infer(visitor);
            
            // obtain the (approximate) argmin
            trbp.arg(labeling);
            //=====================================================
        }
        
        else if (use_TRWS) {
            //=====================================================
            typedef TrbpUpdateRules<Model, opengm::Minimizer> UpdateRules;
            typedef opengm::external::TRWS<Model> TRWS;
            TRWS::Parameter parameter;
            parameter.tolerance_ = 1e-7;
            TRWS trws(gm, parameter);
            
            TRWS::VerboseVisitorType visitor;
            trws.infer(visitor);
            
            // obtain the (approximate) argmin
            trws.arg(labeling);
            //=====================================================
        }
        
        
        else if (use_Gibbs) {
            //=====================================================
            typedef opengm::Gibbs<Model, opengm::Minimizer> Gibbs;
            const size_t numberOfSamplingSteps = 1e4;
            const size_t numberOfBurnInSteps = 1e4;
            Gibbs::Parameter parameter(numberOfSamplingSteps, numberOfBurnInSteps);
            parameter.startPoint_ = labeling;
            Gibbs gibbs(gm, parameter);
            
            Gibbs::VerboseVisitorType visitor;
            gibbs.infer(visitor);
            
            // obtain the (approximate) argmin
            gibbs.arg(labeling);
            //=====================================================
        }
        
        else if (use_SwendsenWang) {
            //===================================================== So slow
            typedef opengm::SwendsenWang<Model, opengm::Minimizer> SwendsenWang;
            SwendsenWang::Parameter parameter;
            parameter.maxNumberOfSamplingSteps_ = 1e4;
            parameter.numberOfBurnInSteps_ = 1e4;
            
            parameter.initialState_ = labeling;
            SwendsenWang sw(gm, parameter);
            
            SwendsenWang::VerboseVisitorType visitor;
            sw.infer(visitor);
            
            // obtain the (approximate) argmin
            sw.arg(labeling);
            //=====================================================
        }
        
        else if (use_DP) {
            //===================================================== So slow
            typedef opengm::DynamicProgramming<Model, opengm::Minimizer> DP;

            DP dp(gm);
            DP::VerboseVisitorType visitor;
            dp.infer(visitor);
            
            // obtain the (approximate) argmin
            dp.arg(labeling);
            //=====================================================
        }
        
        
        else if (use_MQPBO) {
            typedef opengm::MQPBO<Model, opengm::Minimizer> MQPBO;
            // space must be DiscreteSpace, not SimpleDiscreteSpace
            
            MQPBO mqpbo(gm);
            MQPBO::VerboseVisitorType visitor;
            mqpbo.infer(visitor);
            
            // obtain the (approximate) argmin
            mqpbo.arg(labeling);
            
        }
        
        
        
        else if (use_MRFLIB_ICM) {
            //=====================================================
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::ICM;
            para.energyType_ = MRFLIB::Parameter::WEIGHTEDTABLE;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        else if (use_MRFLIB_aexp) {
            //===================================================== Segmentation fault
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::EXPANSION;
//            para.energyType_ = MRFLIB::Parameter::TL1;
            para.energyType_ = MRFLIB::Parameter::TABLES;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        else if (use_MRFLIB_ab) {
            //===================================================== Segmentation fault
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::SWAP;
//            para.energyType_ = MRFLIB::Parameter::TL1;
            para.energyType_ = MRFLIB::Parameter::TABLES;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        else if (use_MRFLIB_LBP) {
            //=====================================================
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::MAXPRODBP;
            para.energyType_ = MRFLIB::Parameter::TABLES;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        else if (use_MRFLIB_BPS) {
            //=====================================================
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::BPS;
            para.energyType_ = MRFLIB::Parameter::VIEW;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        else if (use_MRFLIB_TRWS) {
            //=====================================================
            typedef opengm::external::MRFLIB<Model> MRFLIB;
            
            MRFLIB::Parameter para;
            para.inferenceType_ = MRFLIB::Parameter::TRWS;
            para.energyType_ = MRFLIB::Parameter::VIEW;
            para.numberOfIterations_ = 10;
            MRFLIB mrf(gm,para);
            MRFLIB::VerboseVisitorType visitor;
            mrf.infer(visitor);
            
            // obtain the (approximate) argmin
            mrf.arg(labeling);
            //=====================================================
        }
        
        
        
        else if (use_DD_subgradient) {
            //=====================================================
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
            
            // obtain the (approximate) argmin
            ddsg.arg(labeling);
            //=====================================================
        }
        
        
        else if (use_libdai_BP) {
            typedef external::libdai::Bp<Model, opengm::Minimizer> BP;
            const size_t maxIterations=100;
            const double damping=0.0;
            const double tolerance=0.000001;
            BP::UpdateRule updateRule = BP::PARALL; // Bp::UpdateRule = PARALL | SEQFIX | SEQRND | SEQMAX 9
            size_t verboseLevel=10;
            BP::Parameter parameter(maxIterations, damping, tolerance, updateRule, verboseLevel);
            BP bp(gm, parameter);
            BP::VerboseVisitorType visitor;
            bp.infer(visitor);
            
            // obtain the (approximate) argmin
            bp.arg(labeling);
        }
        else if (use_libdai_TRBP) {
            typedef external::libdai::TreeReweightedBp<Model, opengm::Minimizer> Trbp;
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
            
            // obtain the (approximate) argmin
            trbp.arg(labeling);
        }
        else if (use_libdai_DLGBP) {
            typedef external::libdai::DoubleLoopGeneralizedBP<Model, opengm::Minimizer> DoubleLoopGeneralizedBP;
            
            const bool doubleloop=1;
            // DoubleLoopGeneralizedBP::Clusters=MIN | BETHE | DELTA | LOOP
            const DoubleLoopGeneralizedBP::Clusters clusters = DoubleLoopGeneralizedBP::BETHE;
            const size_t loopdepth = 3;
            // DoubleLoopGeneralizedBP::Init = UNIFORM | RANDOM
            const DoubleLoopGeneralizedBP::Init init = DoubleLoopGeneralizedBP::UNIFORM;
            
            const size_t maxiter=10000;
            const double tolerance=1e-9;
            const size_t verboseLevel=10;
            DoubleLoopGeneralizedBP::Parameter parameter(doubleloop, clusters, loopdepth, init, maxiter, tolerance, verboseLevel);

            DoubleLoopGeneralizedBP gdlbp(gm, parameter);
            DoubleLoopGeneralizedBP::VerboseVisitorType visitor;
            gdlbp.infer(visitor);
            
            // obtain the (approximate) argmin
            gdlbp.arg(labeling);
        }

        else if (use_libdai_FBP) {

            typedef external::libdai::FractionalBp<Model, opengm::Minimizer> FractionalBp;
            
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
            
            // obtain the (approximate) argmin
            fbp.arg(labeling);

        }
        

        
        
        
        
        else if (use_libdai_TEP) {
            
            typedef external::libdai::TreeExpectationPropagation<Model, opengm::Minimizer> TreeExpectationPropagation;
            
            //TreeExpectationPropagation::TreeEpType = ORG | ALT
            TreeExpectationPropagation::TreeEpType treeEpTyp=TreeExpectationPropagation::ORG;
            const size_t maxiter=10000;
            const double tolerance=1e-9;
            size_t verboseLevel=10;
            TreeExpectationPropagation::Parameter parameter(treeEpTyp, maxiter, tolerance, verboseLevel);
            TreeExpectationPropagation treeep(gm, parameter);
            TreeExpectationPropagation::VerboseVisitorType visitor;
            treeep.infer(visitor);
            
            // obtain the (approximate) argmin
            treeep.arg(labeling);
        }
        
        else if (use_libdai_JT) {
            
            typedef external::libdai::JunctionTree<Model, opengm::Minimizer> JunctionTree;
            
            // JunctionTree::UpdateRule = HUGIN | SHSH
            JunctionTree::UpdateRule updateRule = JunctionTree::HUGIN;
            // JunctionTree::Heuristic = MINFILL | WEIGHTEDMINFILL | MINWEIGHT | MINNEIGHBORS
            JunctionTree::Heuristic heuristic = JunctionTree::MINWEIGHT;
            size_t verboseLevel=10;
            JunctionTree::Parameter parameter(updateRule, heuristic,verboseLevel);
            JunctionTree jt(gm, parameter);
            JunctionTree::VerboseVisitorType visitor;
            jt.infer(visitor);
            
            // obtain the (approximate) argmin
            jt.arg(labeling);
        }
        
        
        else if (use_libdai_Gibbs) {
            
            typedef external::libdai::Gibbs<Model, opengm::Minimizer> Gibbs;
            
            const size_t maxiter = 10000;
            const size_t burnin  = 100;
            const size_t restart = 10000;
            const size_t verbose = 10;
            Gibbs::Parameter parameter(maxiter, burnin, restart, verbose);
            Gibbs gibbs(gm, parameter);
            
            Gibbs::VerboseVisitorType visitor;
            gibbs.infer(visitor);
            
            // obtain the (approximate) argmin
            gibbs.arg(labeling);
        }
        
        else if (use_libdai_MF) {  // libc++abi.dylib: terminate called throwing an exception
            
            typedef external::libdai::MeanField<Model, opengm::Minimizer> MeanField;
            
            const size_t maxiter=10000;
            const double damping=0.2;
            const double tolerance=1e-9;
            // MeanField::UpdateRule = NAIVE | HARDSPIN
            const MeanField::UpdateRule updateRule = MeanField::NAIVE;
            // MeanField::Init = UNIFORM | RANDOM
            const MeanField::Init init = MeanField::UNIFORM;
            const size_t verboseLevel=10;
            MeanField::Parameter parameter(maxiter,damping,tolerance,updateRule,init,verboseLevel);
            MeanField mf(gm, parameter);
            
            MeanField::VerboseVisitorType visitor;
            mf.infer(visitor);
            
            // obtain the (approximate) argmin
            mf.arg(labeling);
        }
        
        
        
        
        
        
        Eigen::MatrixXf mygrid0 = Eigen::MatrixXf::Zero(nx,ny);
        Eigen::MatrixXf mygrid1 = Eigen::MatrixXf::Zero(nx,ny);
        Eigen::MatrixXf mygrid2 = Eigen::MatrixXf::Zero(nx,ny);
        std::vector<Eigen::MatrixXf> gridvec2;
        gridvec2.push_back(mygrid0);
        gridvec2.push_back(mygrid1);
        gridvec2.push_back(mygrid2);
        
        
        // output the (approximate) argmin
        size_t variableIndex = 0;
        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                
                gridvec2[labeling[variableIndex]](y,x) = 1;
                
                ++variableIndex;
                
            }
        }
        
        imshow("RGB2", gridvec2, 1.0);
        
        
        
        
        
        
        
        bool isBreak = false;
        int key;
        if( (key = cv::waitKey(0)) >= 0) {
            switch (key)
            {
                case 'q' : isBreak = true; break;
                    
                default: break;
            }
        }
        if (isBreak) break;
        
        
    }
    
    
}
