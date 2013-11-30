//
//  Created by Toru Tamaki on 2013/11/28.
//  Copyright (c) 2013 tamaki. All rights reserved.
//

//
// code taken from
// - opengm/src/examples/image-processing-examples/grid_potts.cxx at http://hci.iwr.uni-heidelberg.de/opengm2/
// - MRF-lib/mrfstereo/mrfstereo.cpp at http://vision.middlebury.edu/MRF/code/
//


#include "opengmtest.h"

void
imshow(const std::string &title,
       const Eigen::MatrixXf &img,
       bool pseudocolor = false)
{
    assert (title != "");
    assert (img.cols() > 0);
    assert (img.rows() > 0);
    
    if(pseudocolor){
        Eigen::MatrixXf angle;
        angle = img*180.;
        cv::Mat _hsv[3], hsv;
        cv::eigen2cv( angle, _hsv[0]);
        _hsv[0] = _hsv[0] + 20;
        _hsv[1] = cv::Mat::ones(_hsv[0].size(), CV_32F);
        _hsv[2] = cv::Mat::ones(_hsv[0].size(), CV_32F);
        
        cv::merge(_hsv, 3, hsv);
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::imshow(title, bgr);
    }else{
        cv::Mat imgCV;
        cv::eigen2cv(img, imgCV);
        cv::imshow(title, imgCV);
    }
}

void
imshow(const std::string &title,
       const std::vector<Eigen::MatrixXf> &gridvec,
       const float maxval)
{
    
    cv::Mat _rgb[3], rgb;
    cv::eigen2cv(gridvec[0], _rgb[0]);
    cv::eigen2cv(gridvec[1], _rgb[1]);
    cv::eigen2cv(gridvec[2], _rgb[2]);
    
    
    cv::merge(_rgb, 3, rgb);
    cv::imshow(title, rgb*(1.0/maxval));
    
}





// model parameters (global variables are used only in example code)
int nx = 100; // width/height of the grid
size_t numberOfLabels = 4; // just for RGB
int lambda = 8; //  x0.1  // coupling strength of the Potts model

// this function maps a node (x, y) in the grid to a unique variable index
inline size_t variableIndex(const size_t x, const size_t y) {
    return x + nx * y;
}







template <class INF>
void
showMarginals(const std::vector<size_t> &labeling,
              const INF &inf,
              const int nx,
              const double sigma)
{
    std::vector<Eigen::MatrixXf> gridvec3;
    for(size_t s = 0; s < numberOfLabels; ++s) {
        Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,nx);
        gridvec3.push_back(mygrid);
    }
    
    Eigen::MatrixXf entropy = Eigen::MatrixXf::Zero(nx,nx);

    for(size_t y = 0; y < nx; ++y)
        for(size_t x = 0; x < nx; ++x) {
            typename INF::IndependentFactorType ift;
            inf.marginal(variableIndex(x, y), ift);
            //std::cerr << "y,x,s " << y << " " << x;
            double sum = 0;
            for(size_t s = 0; s < numberOfLabels; ++s) {
                //std::cerr << " " << ift(s);
//                Estimated label has the least value (that is 0)
//                because now the problem is minimizing,
//                and marginals are not normalized
//                because this is factor graph (undirected), not bayes net (directed).
                double p = std::exp(-ift(s)/sigma);
                gridvec3[s](y,x) = p;
                sum += p;
                
                entropy(y,x) += (-p * std::log(p));
            }
            for(size_t s = 0; s < numberOfLabels; ++s) {
                gridvec3[s](y,x) /= sum;
                //std::cerr << " " << gridvec3[s](y,x);
            }
            //std::cerr << " " << entropy(y,x);
            //std::cerr << std::endl;
        }
    double maxval = entropy.maxCoeff();
    double minval = entropy.minCoeff();
    entropy.array() -= minval;
    entropy *= (maxval - minval);
    
    imshow("marginal", gridvec3, 1.0);
    imshow("entropy", entropy);
}




int main() {
    
    
    
    
    cv::namedWindow("control panel", CV_WINDOW_NORMAL | CV_GUI_EXPANDED);
    
    cv::createTrackbar("nx", "control panel", &nx, 300,  NULL);
    cv::createTrackbar("lambda", "control panel", &lambda, 10,  NULL);
    

#include "controlpanel.hxx"
    
    
    int noise = 2;
    cv::createTrackbar("noise*10", "control panel", &noise, 10,  NULL);
    int recreate = 1;
    cv::createTrackbar("re-create", "control panel", &recreate, 1,  NULL);

    int sigma = 1;
    cv::createTrackbar("sigma", "control panel", &sigma, 100,  NULL);

    
    
    
    std::vector<Eigen::MatrixXf> gridvec;
    
    
    while(1){
        
        
       
        // construct a label space with
        // - nx * nx variables
        // - each having numberOfLabels many labels
//      typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;  // mqpbo does not accept this.
        typedef opengm::DiscreteSpace<size_t, size_t> Space;
        Space space((size_t)(nx * nx), numberOfLabels);
        
        if(recreate == 1){
            gridvec.clear();
            double cx[] = {nx/4, nx/4*3, nx/4*3};
            double cy[] = {nx/4, nx/4,   nx/4*3};
            Eigen::MatrixXf mygridsum = Eigen::MatrixXf::Zero(nx,nx);
            for(size_t s = 0; s < numberOfLabels-1; ++s) {
                Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,nx);
                for(size_t y = 0; y < nx; ++y)
                    for(size_t x = 0; x < nx; ++x) {
                        double p;
                        p = 0.8 * exp( - ((cx[s] - x)*(cx[s] - x) + (cy[s] - y)*(cy[s] - y)) / (nx*2) );
                        p += noise/10. * (rand() / (double)RAND_MAX);
                        mygrid(y,x) = p;
                    }
                mygridsum += mygrid;
                gridvec.push_back(mygrid);
            }
            {
                Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,nx);
                for(size_t y = 0; y < nx; ++y)
                    for(size_t x = 0; x < nx; ++x) {
                        mygrid(y,x) = 0.2;
                        //mygrid(y,x) = 0.5 * (rand() / (double)RAND_MAX);
//                        mygrid(y,x) = 0;
//                        for(size_t s = 0; s < numberOfLabels-1; ++s) {
//                            double p = gridvec[s](y,x);
//                            mygrid(y,x) += -p * std::log(p);
//                        }
//                        mygrid(y,x) *= 0.5;
                    }
                mygridsum += mygrid;
                gridvec.push_back(mygrid);

            }
            for(size_t s = 0; s < numberOfLabels; ++s) {
                gridvec[s] = gridvec[s].cwiseQuotient(mygridsum);
            }
//            cv::setTrackbarPos("re-create", "control panel", 0);
        }
        imshow("RGB", gridvec, 1.0);
        
        
        // construct a graphical model with
        // - addition as the operation (template parameter Adder)
        // - support for Potts functions (template parameter PottsFunction<double>)
        typedef opengm::GraphicalModel<double, opengm::Adder,
                                       OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>,
                                                         opengm::PottsFunction<double>),
                                       Space> Model;
        Model gm(space);
        
        // for each node (x, y) in the grid, i.e. for each variable
        // variableIndex(x, y) of the model, add one 1st order functions
        // and one 1st order factor
        for(size_t y = 0; y < nx; ++y)
            for(size_t x = 0; x < nx; ++x) {
                // function
                const size_t shape[] = {numberOfLabels};
                opengm::ExplicitFunction<double> f(shape, shape + 1);
                for(size_t s = 0; s < numberOfLabels; ++s) {
                    double tmp = gridvec[s](y,x);
                    f(s) = -log(tmp);
                }
                Model::FunctionIdentifier fid = gm.addFunction(f);
                
                // factor
                size_t variableIndices[] = {variableIndex(x, y)};
                gm.addFactor(fid, variableIndices, variableIndices + 1);
            }
        
        // add one (!) 2nd order Potts function
#ifdef WITH_FASTPD
        const double valEqual = 0.; // FastPD requires this.
#else
        const double valEqual = -log(lambda/10.);
#endif
        const double valUnequal = -log( (1.0-lambda/10.) / 2);
        opengm::PottsFunction<double> f(numberOfLabels, numberOfLabels, valEqual, valUnequal);
        Model::FunctionIdentifier fid = gm.addFunction(f);
        
        // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
        // add one factor that connects the corresponding variable indices and
        // refers to the Potts function
        for(size_t y = 0; y < nx; ++y)
            for(size_t x = 0; x < nx; ++x) {
                if(x + 1 < nx) { // (x, y) -- (x + 1, y)
                    size_t variableIndices[] = {variableIndex(x, y), variableIndex(x + 1, y)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
                if(y + 1 < nx) { // (x, y) -- (x, y + 1)
                    size_t variableIndices[] = {variableIndex(x, y), variableIndex(x, y + 1)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
            }

        

        std::vector<size_t> labeling(nx * nx);
        
#include "infer.hxx"
        
        
        
        std::vector<Eigen::MatrixXf> gridvec2;
        for(size_t s = 0; s < numberOfLabels; ++s) {
            Eigen::MatrixXf mygrid = Eigen::MatrixXf::Zero(nx,nx);
            gridvec2.push_back(mygrid);
        }
        
        
        // output the (approximate) argmin
        size_t variableIndex = 0;
        for(size_t y = 0; y < nx; ++y) {
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
