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
        Opt.filenameR = vm["R"].as<string>();
    else{
        cout << "no right image filename is specified." << endl
        << desc << endl;
        exit(1);
    }
    
    return Opt;
}

template <class INF>
void
showMarginals(const std::vector<size_t> &labeling,
              const INF &inf,
              const int nx,
              const int sigma)
{
    // empty
}

int main ( int argc, char **argv )
{
    
    options Option = parseOptions(argc, argv);

    
    cv::namedWindow("control panel", CV_WINDOW_NORMAL | CV_GUI_EXPANDED);
    
    int lambda = 20; // weight for the pairwise term
    cv::createTrackbar("lambda", "control panel", &lambda, 100,  NULL);
    

#include "controlpanel.hxx"
  
    
    
    int nD = 16; // number of disparities
    cv::createTrackbar("nD", "control panel", &nD, 50,  NULL);

    int sigma = 10; // unused now;
    
    
    
    std::vector<Eigen::MatrixXf> gridvec;
    
    
    
    cv::Mat imgL = cv::imread(Option.filenameL);
    cv::Mat imgR = cv::imread(Option.filenameR);
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
    assert(imgL.size() == imgR.size());
    
    vector<cv::Mat> imgLs, imgRs;
    cv::split(imgL, imgLs);
    cv::split(imgR, imgRs);

    cv::waitKey(0);
    
    while(1){
        

        
        
        // construct a label space with
        // - nx * ny variables
        // - each having numberOfLabels many labels
//      typedef SimpleDiscreteSpace<size_t, size_t> Space;  // mqpbo does not accept this.
        typedef opengm::DiscreteSpace<size_t, size_t> Space;
        Space space(nx * ny, (size_t)nD);

        // construct a graphical model with
        // - addition as the operation (template parameter Adder)
        // - support for Potts functions (template parameter PottsFunction<double>)
        typedef opengm::GraphicalModel<double, opengm::Adder,
                                      OPENGM_TYPELIST_2(opengm::ExplicitFunction<double>,
                                                        opengm::TruncatedAbsoluteDifferenceFunction<double> ) ,
                                       Space> Model;
        Model gm(space);
        
        

        
        
        ////////////////////////////////////////// from mrfstereo.cpp of MRF-Lib
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
                
                const size_t shape[] = {(int)nD};
                opengm::ExplicitFunction<double> f(shape, shape + 1);

                
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
                                int im2c = imgRs[b].at<uchar>(y,x2);
                                int im2l = x2 == 0?    im2c : (im2c + imgRs[b].at<uchar>(y,x2-1)) / 2;
                                int im2r = x2 == nx-1? im2c : (im2c + imgRs[b].at<uchar>(y,x2+1)) / 2;
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
                                int di = imgLs[b].at<uchar>(y,x) - imgRs[b].at<uchar>(y,x2);
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
                    
//                    if (y == 150) {
//                        DSIy.at<uchar>(d,x) = f(d); // check DSI at a particular y
//                    }
                }
                
                // factor
                Model::FunctionIdentifier fid = gm.addFunction(f);
                size_t variableIndices[] = {variableIndex(x, y, nx)};
                gm.addFactor(fid, variableIndices, variableIndices + 1);
                
                
            }
        }
        
        //////////////////////////////////////////
        

        opengm::TruncatedAbsoluteDifferenceFunction<double> f(nD, nD, 2, lambda);
        Model::FunctionIdentifier fid = gm.addFunction(f);
        
        // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
        // add one factor that connects the corresponding variable indices and
        // refers to the Potts function
        for(size_t y = 0; y < ny; ++y)
            for(size_t x = 0; x < nx; ++x) {
                if(x + 1 < nx) { // (x, y) -- (x + 1, y)
                    size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x + 1, y, nx)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
                if(y + 1 < ny) { // (x, y) -- (x, y + 1)
                    size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x, y + 1, nx)};
                    std::sort(variableIndices, variableIndices + 2);
                    gm.addFactor(fid, variableIndices, variableIndices + 2);
                }
            }


        std::vector<size_t> labeling(nx * ny);
        
#include "infer.hxx"
        
        
        
        cv::Mat labels(ny, nx, CV_8U);
        
        
        
        // output the (approximate) argmin
        for(size_t y = 0; y < ny; ++y) {
            for(size_t x = 0; x < nx; ++x) {
                
                labels.at<uchar>(y,x) = labeling[variableIndex(x, y, nx)] * (255/(nD-1));
                
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
