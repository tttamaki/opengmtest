//
//  Created by Toru Tamaki on 2013/11/28.
//  Copyright (c) 2013 tamaki. All rights reserved.
//

#ifndef opengmtest_opengmtest_h
#define opengmtest_opengmtest_h

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
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

#include <opengm/functions/potts.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>


#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>


#ifdef WITH_SAMPLING
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/swendsenwang.hxx>
#endif



#ifdef WITH_MAXFLOW
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

#ifdef WITH_MAXFLOW_IBFS
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutibfs.hxx>
#endif

#ifdef WITH_BOOST
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif


#ifdef WITH_AD3
#include <opengm/inference/loc.hxx>
#endif


#ifdef WITH_QPBO
#include <opengm/inference/mqpbo.hxx>
#include <opengm/inference/external/qpbo.hxx>
#include <opengm/inference/alphaexpansionfusion.hxx>
#endif

#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif

#ifdef WITH_MRFLIB
#include <opengm/inference/external/mrflib.hxx>
#endif


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

#if defined(WITH_QPBO) && defined(WITH_AD3)
#include <opengm/inference/auxiliary/fusion_move/fusion_mover.hxx>
#endif


#endif
