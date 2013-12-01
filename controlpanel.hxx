//
//  Created by Toru Tamaki on 2013/11/28.
//  Copyright (c) 2013 tamaki. All rights reserved.
//

#ifndef opengmtest_controlpanel_hxx
#define opengmtest_controlpanel_hxx


int use_BP = 1;
cv::createTrackbar("BP", "control panel", &use_BP, 1,  NULL);

#ifdef WITH_MAXFLOW
int use_ab_graphcut = 0;
cv::createTrackbar("a-b GC (Kol)", "control panel", &use_ab_graphcut, 1,  NULL);
int use_aexp_graphcut = 0;
cv::createTrackbar("a-exp GC (Kol)", "control panel", &use_aexp_graphcut, 1,  NULL);
#endif

#ifdef WITH_MAXFLOW_IBFS
int use_ab_graphcut_ibfs = 0;
cv::createTrackbar("a-b GC (IBFS)", "control panel", &use_ab_graphcut_ibfs, 1,  NULL);
int use_aexp_graphcut_ibfs = 0;
cv::createTrackbar("a-exp GC (IBFS)", "control panel", &use_aexp_graphcut_ibfs, 1,  NULL);
#endif

#ifdef WITH_BOOST
int use_ab_graphcut_boost = 0;
cv::createTrackbar("a-b GC (Boost)", "control panel", &use_ab_graphcut_boost, 1,  NULL);
int use_aexp_graphcut_boost = 0;
cv::createTrackbar("a-exp GC (Boost)", "control panel", &use_aexp_graphcut_boost, 1,  NULL);
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




#endif
