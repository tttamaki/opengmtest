//
//  Created by Toru Tamaki on 2013/11/28.
//  Copyright (c) 2013 tamaki. All rights reserved.
//

#ifndef opengmtest_controlpanel_hxx
#define opengmtest_controlpanel_hxx


#define ADD_PANEL(COND,EXP,VAR) \
int VAR = 0; if(COND){ cv::createTrackbar(EXP, "control panel", &VAR, 1,  NULL); }



ADD_PANEL(true, "BP", use_BP)

#ifdef WITH_MAXFLOW
ADD_PANEL(Opt.MAXFLOW, "a-b GC (Kol)", use_ab_graphcut)
ADD_PANEL(Opt.MAXFLOW, "a-exp GC (Kol)", use_aexp_graphcut)
#endif

#ifdef WITH_MAXFLOW_IBFS
ADD_PANEL(Opt.MAXFLOW_IBFS, "a-b GC (IBFS)",   use_ab_graphcut_ibfs)
ADD_PANEL(Opt.MAXFLOW_IBFS, "a-exp GC (IBFS)", use_aexp_graphcut_ibfs)
#endif

#ifdef WITH_BOOST
ADD_PANEL(Opt.BOOST, "a-b GC (Boost KOL)",   use_ab_graphcut_boost_kl)
ADD_PANEL(Opt.BOOST, "a-exp GC (Boost KOL)", use_aexp_graphcut_boost_kl)
ADD_PANEL(Opt.BOOST, "a-b GC (Boost ED)",    use_ab_graphcut_boost_ed)
ADD_PANEL(Opt.BOOST, "a-exp GC (Boost ED)",  use_aexp_graphcut_boost_ed)
ADD_PANEL(Opt.BOOST, "a-b GC (Boost PR)",    use_ab_graphcut_boost_pr)
ADD_PANEL(Opt.BOOST, "a-exp GC (Boost PR)",  use_aexp_graphcut_boost_pr)
#endif

ADD_PANEL(true, "ICM", use_icm)
ADD_PANEL(true, "LazyFlipper", use_lazyFlipper)

#ifdef WITH_AD3
ADD_PANEL(Opt.AD3, "LOC", use_loc)
#endif

ADD_PANEL(true, "TRBP", use_TRBP)


#ifdef WITH_TRWS
ADD_PANEL(Opt.TRWS, "TRWS", use_TRWS)
#endif

#ifdef WITH_SAMPLING
ADD_PANEL(Opt.SAMPLING, "Gibbs", use_Gibbs)
ADD_PANEL(Opt.SAMPLING, "SW",    use_SwendsenWang)
#endif

ADD_PANEL(false, "DP", use_DP)

#ifdef WITH_QPBO
ADD_PANEL(Opt.QPBO, "MQPBO", use_MQPBO)
ADD_PANEL(Opt.QPBO, "a-exp-fusion", use_aexp_fusion)
#endif

#ifdef WITH_MRFLIB
ADD_PANEL(Opt.MRFLIB, "MRFLIB ICM",   use_MRFLIB_ICM)
ADD_PANEL(Opt.MRFLIB, "MRFLIB a-exp", use_MRFLIB_aexp)
ADD_PANEL(Opt.MRFLIB, "MRFLIB ab",    use_MRFLIB_ab)
ADD_PANEL(Opt.MRFLIB, "MRFLIB LBP",   use_MRFLIB_LBP)
ADD_PANEL(Opt.MRFLIB, "MRFLIB BPS",   use_MRFLIB_BPS)
ADD_PANEL(Opt.MRFLIB, "MRFLIB TRWS",  use_MRFLIB_TRWS)
#endif

ADD_PANEL(true, "DD subgrad", use_DD_subgradient)

#ifdef WITH_LIBDAI
ADD_PANEL(Opt.LIBDAI, "libdai BP",    use_libdai_BP)
ADD_PANEL(Opt.LIBDAI, "libdai TRBP",  use_libdai_TRBP)
ADD_PANEL(Opt.LIBDAI, "libdai DLGBP", use_libdai_DLGBP)
ADD_PANEL(Opt.LIBDAI, "libdai FBP",   use_libdai_FBP)
ADD_PANEL(Opt.LIBDAI, "libdai JT",    use_libdai_JT)
ADD_PANEL(Opt.LIBDAI, "libdai TEP",   use_libdai_TEP)
ADD_PANEL(Opt.LIBDAI, "libdai Gibbs", use_libdai_Gibbs)
ADD_PANEL(Opt.LIBDAI, "libdai MF",    use_libdai_MF)
#endif

#ifdef WITH_FASTPD
ADD_PANEL(Opt.FASTPD, "FastPD", use_fastPD)
#endif

#ifdef WITH_MPLP
ADD_PANEL(Opt.MPLP, "MPLP", use_MPLP)
#endif

#ifdef WITH_GCO
ADD_PANEL(Opt.GCO, "GCO aexp", use_GCO_aexp)
ADD_PANEL(Opt.GCO, "GCO swap", use_GCO_swap)
#endif




#endif
