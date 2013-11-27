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
#include <cmath>

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
#include <opengm/operations/Minimizer.hxx>


#ifdef WITH_MAXFLOW
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>

#ifdef WITH_AD3
#include <opengm/inference/loc.hxx>
#endif

#include <opengm/inference/dynamicprogramming.hxx>

#ifdef WITH_QPBO
#include <opengm/inference/mqpbo.hxx>
#include <opengm/inference/external/qpbo.hxx>
#include <opengm/inference/alphaexpansionfusion.hxx>
#endif

#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif

#ifdef WITH_MRF
#include <opengm/inference/external/mrflib.hxx>
#endif

#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>

#ifdef WITH_LIBDAI
#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_expectation_propagation.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/gibbs.hxx>
#include <opengm/inference/external/libdai/mean_field.hxx>
#endif

#ifdef WITH_FASTPD
#undef MAX
#undef MIN
#include <opengm/inference/external/fastPD.hxx> // this defines functions MAX and MIN which conflict with MAX/MIN macros defined in opencv2
#endif


#ifdef WITH_MPLP
#include <opengm/inference/external/mplp.hxx>
#undef eps
#endif

#ifdef WITH_GCO
#include <opengm/inference/external/gco.hxx> // can't be complied with WITH_MRF at the same time because two files of the same name "GCoptimization.h" conflicts to each other.
#endif


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
    cv::imshow(title, rgb*(1.0/maxval));
    
}





//using namespace std; // 'using' is used only in example code
using namespace opengm;

// model parameters (global variables are used only in example code)
//int nx = 100; // width/height of the grid
//size_t numberOfLabels = 3; // just for RGB
int lambda = 8; //  x0.1  // coupling strength of the Potts model

// this function maps a node (x, y) in the grid to a unique variable index
inline size_t variableIndex(const size_t x, const size_t y, const size_t nx) {
    return x + nx * y;
}






#include "boost/program_options.hpp"

struct options {
    std::string filenameL;
    std::string filenameR;
};


options parseOptions(int argc, char* argv[]) {
    
    namespace po = boost::program_options;
    
    po::options_description desc("Options");
    desc.add_options()
    ("help", "This help message.")
    ("L", po::value<string>(), "left  image filename")
    ("R", po::value<string>(), "right image filename")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    options Opt;
    
    if (vm.count("help")) {
        cout << desc << endl;
        exit(0);
    }
    
    if (vm.count("L"))
        Opt.filenameL = vm["L"].as<string>();
    else{
        cout << "no left  image filename is specified." << endl
        << desc << endl;
        exit(1);
    }
    
    if (vm.count("R"))
        Opt.filenameL = vm["R"].as<string>();
    else{
        cout << "no right image filename is specified." << endl
        << desc << endl;
        exit(1);
    }
    
    return Opt;
}


int main ( int argc, char **argv )
{
    
    options Option = parseOptions(argc, argv);

    
    cv::namedWindow("control panel", CV_WINDOW_NORMAL | CV_GUI_EXPANDED);
    
    //cv::createTrackbar("nx", "control panel", &nx, 300,  NULL);
    cv::createTrackbar("lambda", "control panel", &lambda, 10,  NULL);
    

    int use_BP = 1;
    cv::createTrackbar("BP", "control panel", &use_BP, 1,  NULL);
    
#ifdef WITH_MAXFLOW
    int use_ab_graphcut = 0;
    cv::createTrackbar("a-b GC", "control panel", &use_ab_graphcut, 1,  NULL);
    int use_aexp_graphcut = 0;
    cv::createTrackbar("a-exp GC", "control panel", &use_aexp_graphcut, 1,  NULL);
#endif
    
    int use_icm = 0;
    cv::createTrackbar("ICM", "control panel", &use_icm, 1,  NULL);
    int use_lazyFlipper = 0;
    cv::createTrackbar("LazyFlipper", "control panel", &use_lazyFlipper, 1,  NULL);
#ifdef WITH_AD3
    int use_loc = 0;
    cv::createTrackbar("LOC", "control panel", &use_loc, 1,  NULL);
#endif
    
    int use_TRBP = 0;
    cv::createTrackbar("TRBP", "control panel", &use_TRBP, 1,  NULL);
    
#ifdef WITH_TRWS
    int use_TRWS = 0;
    cv::createTrackbar("TRWS", "control panel", &use_TRWS, 1,  NULL);
#endif

#ifdef WITH_SAMPLING
    // Gibbs and SW seem to be implemented for minimizer only.
    int use_Gibbs = 0;
    cv::createTrackbar("Gibbs", "control panel", &use_Gibbs, 1,  NULL);
    int use_SwendsenWang = 0;
    cv::createTrackbar("SW", "control panel", &use_SwendsenWang, 1,  NULL);
#endif
    
    int use_DP = 0;
    cv::createTrackbar("DP", "control panel", &use_DP, 1,  NULL);
#ifdef WITH_QPBO
    int use_MQPBO = 0;
    cv::createTrackbar("MQPBO", "control panel", &use_MQPBO, 1,  NULL);
    int use_aexp_fusion = 0;
    cv::createTrackbar("a-exp-fusion", "control panel", &use_aexp_fusion, 1,  NULL);
#endif

#ifdef WITH_MRF
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
#endif
    int use_DD_subgradient = 0;
    cv::createTrackbar("DD subgrad", "control panel", &use_DD_subgradient, 1,  NULL);

#ifdef WITH_LIBDAI
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
#endif
    
#ifdef WITH_FASTPD
    int use_fastPD = 0;
    cv::createTrackbar("FastPD", "control panel", &use_fastPD, 1,  NULL);
#endif
    
#ifdef WITH_MPLP
    int use_MPLP = 0;
    cv::createTrackbar("MPLP", "control panel", &use_MPLP, 1,  NULL);
#endif
    
#ifdef WITH_GCO
    int use_GCO_aexp = 0;
    cv::createTrackbar("GCO aexp", "control panel", &use_GCO_aexp, 1,  NULL);
    int use_GCO_swap = 0;
    cv::createTrackbar("GCO swap", "control panel", &use_GCO_swap, 1,  NULL);
#endif
    
    
    
    size_t nD = 16; // number of disparities

    
    
//    int noise = 2;
//    cv::createTrackbar("noise*10", "control panel", &noise, 10,  NULL);
//    int recreate = 1;
//    cv::createTrackbar("re-create", "control panel", &recreate, 1,  NULL);

    
    
    
    std::vector<Eigen::MatrixXf> gridvec;
    
    
    
    cv::Mat imgL = cv::imread(Option.filenameL);
    cv::Mat imgR = cv::imread(Option.filenameL);
    int nChannelsL = imgL.channels();
    int nChannelsR = imgR.channels();
    assert(nChannelsL == nChannelsR);
    std::cerr << "nChannelsL " << nChannelsL << std::endl;
    std::cerr << "nChannelsR " << nChannelsR << std::endl;
    std::cerr << "WxH " << imgL.size().width << " " << imgL.size().height << std::endl;
    imshow("left image", imgL);
    imshow("right image", imgR);
    
    size_t nx = imgL.size().width;
    size_t ny = imgL.size().height;
    assert(nx > 0);
    assert(ny > 0);
    
    vector<cv::Mat> imgLs, imgRs;
    cv::split(imgL, imgLs);
    cv::split(imgR, imgRs);

    
    while(1){
        

        
        
        // construct a label space with
        // - nx * nx variables
        // - each having numberOfLabels many labels
//      typedef SimpleDiscreteSpace<size_t, size_t> Space;  // mqpbo does not accept this.
        typedef DiscreteSpace<size_t, size_t> Space;
        Space space(nx * ny, nD);

        // construct a graphical model with
        // - addition as the operation (template parameter Adder)
        // - support for Potts functions (template parameter PottsFunction<double>)
        typedef GraphicalModel<double, Adder,
                               OPENGM_TYPELIST_2(ExplicitFunction<double> , PottsFunction<double> ) ,
                               Space> Model;
        Model gm(space);
        
        
        
        
//        if(recreate == 1){
//            gridvec.clear();
//            double c[] = {nx/4, nx/2, nx/4*3};
//            Eigen::MatrixXf mygridsum = Eigen::MatrixXf::Zero(nx,nx);
//            for(size_t s = 0; s < numberOfLabels; ++s) {
//                Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,nx);
//                for(size_t y = 0; y < nx; ++y)
//                    for(size_t x = 0; x < nx; ++x) {
//                        double p;
//                        p = 0.8 * exp( - ((c[s] - x)*(c[s] - x) + (c[s] - y)*(c[s] - y)) / (nx*2) );
//                        p += noise/10. * (rand() / (double)RAND_MAX);
//                        mygrid(y,x) = p;
//                    }
//                mygridsum += mygrid;
//                gridvec.push_back(mygrid);
//            }
//            for(size_t s = 0; s < numberOfLabels; ++s) {
//                gridvec[s] = gridvec[s].cwiseQuotient(mygridsum);
//            }
////            cv::setTrackbarPos("re-create", "control panel", 0);
//        }
//        imshow("RGB", gridvec, 1.0);
//        
//        
//
//        
//        // for each node (x, y) in the grid, i.e. for each variable
//        // variableIndex(x, y) of the model, add one 1st order functions
//        // and one 1st order factor
//        for(size_t y = 0; y < nx; ++y)
//            for(size_t x = 0; x < nx; ++x) {
//                // function
//                const size_t shape[] = {numberOfLabels};
//                ExplicitFunction<double> f(shape, shape + 1);
//                for(size_t s = 0; s < numberOfLabels; ++s) {
//                    double tmp = gridvec[s](y,x);
//                    f(s) = -log(tmp);
//                }
//                Model::FunctionIdentifier fid = gm.addFunction(f);
//                
//                // factor
//                size_t variableIndices[] = {variableIndex(x, y)};
//                gm.addFactor(fid, variableIndices, variableIndices + 1);
//            }
        
        
        
        ////////////////////////////////////////// from mrfstereo of MRF-Lib
        int birchfield   = 1; // use Birchfield/Tomasi costs
        int squaredDiffs = 0; // use squared differences
        int truncDiffs = 255; // truncated differences

        int nColors = std::min(3, nChannelsL);
        
        // worst value for sumdiff below
        int worst_match = nColors * (squaredDiffs ? 255 * 255 : 255);
        // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
        int maxsumdiff = squaredDiffs ? truncDiffs * truncDiffs : nColors * abs(truncDiffs);
        // value for out-of-bounds matches
        int badcost = std::min(worst_match, maxsumdiff);
        
        int dsiIndex = 0;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                //uchar *pix1 = &im1.Pixel(x, y, 0);
                
                const size_t shape[] = {nD};
                ExplicitFunction<double> f(shape, shape + 1);

                
                for (int d = 0; d < nD; d++) {
                    int x2 = x-d;
                    int dsiValue;
                    
                    if (x2 >= 0 && d < nD) { // in bounds
                        //uchar *pix2 = &im2.Pixel(x2, y, 0);
                        int sumdiff = 0;
                        
                       
                        for (size_t b = 0; b < nColors; b++) {
                            int diff = 0;
                            if (birchfield) {
                                // Birchfield/Tomasi cost
                                int im1c = imgLs[b].at<uchar>(y,x);
                                int im1l = x == 0?    im1c : (im1c + imgLs[b].at<uchar>(y,x-1)) / 2;
                                int im1r = x == nx-1? im1c : (im1c + imgLs[b].at<uchar>(y,x+1)) / 2;
                                int im2c = imgRs[b].at<uchar>(y,x);
                                int im2l = x2 == 0?    im2c : (im2c + imgRs[b].at<uchar>(y,x+1)) / 2;
                                int im2r = x2 == nx-1? im2c : (im2c + imgRs[b].at<uchar>(y,x+1)) / 2;
                                int min1 = std::min(im1c, std::min(im1l, im1r));
                                int max1 = std::max(im1c, std::max(im1l, im1r));
                                int min2 = std::min(im2c, std::min(im2l, im2r));
                                int max2 = std::max(im2c, std::max(im2l, im2r));
                                int di1 = std::max(0, std::max(im1c - max2, min2 - im1c));
                                int di2 = std::max(0, std::max(im2c - max1, min1 - im2c));
                                diff = std::min(di1, di2);
                            } else {
                                // simple absolute difference
                                //int di = pix1[b] - pix2[b];
                                int di = imgLs[b].at<uchar>(y,x) - imgRs[b].at<uchar>(y,x);
                                diff = abs(di);
                            }
                            // square diffs if requested (Birchfield too...)
                            sumdiff += (squaredDiffs ? diff * diff : diff);
                        }
                        // truncate diffs
                        //dsiValue = std::min(sumdiff, maxsumdiff);
                        f(d) = std::min(sumdiff, maxsumdiff);
                    } else { // out of bounds: use maximum truncated cost
                        //dsiValue = badcost;
                        f(d) = badcost;
                    }
                    
                    // The cost of pixel p and label l is stored at dsi[p*nLabels+l]
                    //dsi[dsiIndex++] = dsiValue;
                    
                   
                }
                
                // factor
                Model::FunctionIdentifier fid = gm.addFunction(f);
                size_t variableIndices[] = {variableIndex(x, y, nx)};
                gm.addFactor(fid, variableIndices, variableIndices + 1);
                
                
            }
        }
        
        //////////////////////////////////////////
        
        

    
        
        
        // add one (!) 2nd order Potts function
#ifdef WITH_FASTPD
        const double valEqual = 0.; // FastPD requires this.
#else
        const double valEqual = -log(lambda/10.);
#endif
        const double valUnequal = -log( (1.0-lambda/10.) / 2);
        PottsFunction<double> f(nD, nD, valEqual, valUnequal);
        Model::FunctionIdentifier fid = gm.addFunction(f);
        
        // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
        // add one factor that connects the corresponding variable indices and
        // refers to the Potts function
        for(size_t y = 0; y < nx; ++y)
            for(size_t x = 0; x < nx; ++x) {
                if(x + 1 < nx) { // (x, y) -- (x + 1, y)
                    size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x + 1, y, nx)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
                if(y + 1 < nx) { // (x, y) -- (x, y + 1)
                    size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x, y + 1, nx)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
            }

        cv::waitKey(0);
        exit(1);

        std::vector<size_t> labeling(nx * nx);
        
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

#ifdef WITH_MAXFLOW
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
#endif
        
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
        
#ifdef WITH_AD3
        else if (use_loc) {
            //=====================================================
            typedef opengm::LOC<Model, opengm::Minimizer> LOC;
            LOC::Parameter parameter;
            parameter.phi_ = 5;
//            parameter.maxRadius_ = 10; // obsolated?
            parameter.maxIterations_ = 100;
            
            LOC loc(gm, parameter);
            LOC::VerboseVisitorType visitor;
            loc.infer(visitor);
            
            // obtain the (approximate) argmin
            loc.arg(labeling);
            //=====================================================
        }
#endif
        
        
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

#ifdef WITH_TRWS
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
#endif

#ifdef WITH_SAMPLING
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
#endif
        
#ifdef WITH_DP
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
#endif
        
#ifdef WITH_QPBO
        else if (use_MQPBO) {
            typedef opengm::MQPBO<Model, opengm::Minimizer> MQPBO;
            // space must be DiscreteSpace, not SimpleDiscreteSpace
            
            MQPBO mqpbo(gm);
            MQPBO::VerboseVisitorType visitor;
            mqpbo.infer(visitor);
            
            // obtain the (approximate) argmin
            mqpbo.arg(labeling);
            
        }
        else if (use_aexp_fusion) {
            typedef opengm::AlphaExpansionFusion<Model, opengm::Minimizer> AlphaExpFusion;
            
            AlphaExpFusion aexp_fusion(gm);
            AlphaExpFusion::VerboseVisitorType visitor;
            aexp_fusion.infer(visitor);
            
            // obtain the (approximate) argmin
            aexp_fusion.arg(labeling);
            
        }
#endif
        
#ifdef WITH_MRF
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
            para.energyType_ = MRFLIB::Parameter::TL2;
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
            para.energyType_ = MRFLIB::Parameter::TL2;
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
#endif
        
        
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
        
#ifdef WITH_FASTPD
        else if (use_fastPD) {
            //=====================================================
            typedef opengm::external::FastPD<Model> fastPD;
            fastPD::Parameter param;
            param.numberOfIterations_ = 1000;

            fastPD fastpd(gm, param);
            fastPD::VerboseVisitorType visitor;
            fastpd.infer(visitor);
            
            // obtain the (approximate) argmin
            fastpd.arg(labeling);
            //=====================================================
        }
#endif

#ifdef WITH_MPLP
        else if (use_MPLP) {
            //=====================================================
            typedef opengm::external::MPLP<Model> Mplp;
            Mplp::Parameter param;
            param.maxTime_ = 120;
            
            Mplp mplp(gm, param);
            Mplp::VerboseVisitorType visitor;
            mplp.infer(visitor);
            
            // obtain the (approximate) argmin
            mplp.arg(labeling);
            //=====================================================
        }
#endif
        
#ifdef WITH_GCO
        else if (use_GCO_aexp) {
            //=====================================================
            typedef opengm::external::GCOLIB<Model> GCO;
            GCO::Parameter param;
            param.inferenceType_ = GCO::Parameter::EXPANSION; // EXPANSION, SWAP
            param.energyType_ = GCO::Parameter::VIEW; // VIEW, TABLES, WEIGHTEDTABLE
            param.useAdaptiveCycles_ = false; // for alpha-expansion
            
            GCO gco(gm, param);
            GCO::VerboseVisitorType visitor;
            gco.infer(visitor);
            
            // obtain the (approximate) argmin
            gco.arg(labeling);
            //=====================================================
        }
        else if (use_GCO_swap) {
            //=====================================================
            typedef opengm::external::GCOLIB<Model> GCO;
            GCO::Parameter param;
            param.inferenceType_ = GCO::Parameter::SWAP; // EXPANSION, SWAP
            param.energyType_ = GCO::Parameter::VIEW; // VIEW, TABLES, WEIGHTEDTABLE
            //param.useAdaptiveCycles_ = false; // for alpha-expansion
            
            GCO gco(gm, param);
            GCO::VerboseVisitorType visitor;
            gco.infer(visitor);
            
            // obtain the (approximate) argmin
            gco.arg(labeling);
            //=====================================================
        }
#endif
     
        
#ifdef WITH_LIBDAI
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
#endif
        
        
        
        
        
        cv::Mat labels(nx, ny, CV_8U);
        
        
        
        // output the (approximate) argmin
        for(size_t y = 0; y < nx; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                
                labels.at<uchar>(y,x) = labeling[variableIndex(x, y, nx)];
                
            }
        }
        
        imshow("labels", labels);
        
        
        
        
        
        
        
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
