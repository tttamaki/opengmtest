OPENGM_EXTSRC = /Users/tamaki/dev/opengm/src/external
OPENGM_EXTLIB = /Users/tamaki/dev/opengm/build/src/external
LIBDAI_LIB    = /Users/tamaki/dev/libdai/lib
EXTLIBS = 

WITH_LIBDAI =
WITH_MAXFLOW = 1
WITH_QPBO = 1
WITH_TRWS = 1
WITH_AD3 = 1
WITH_MRF = 1

ifdef WITH_LIBDAI
  EXTLIBS += -L$(LIBDAI_LIB) -ldai -lgmp -lgmpxx -DWITH_LIBDAI
endif
ifdef WITH_MAXFLOW
  EXTLIBS += -I$(OPENGM_EXTSRC)/MaxFlow-v3.02.src-patched/ -lexternal-library-maxflow -DWITH_MAXFLOW
endif
ifdef WITH_QPBO
  EXTLIBS += -I$(OPENGM_EXTSRC)/QPBO-v1.3.src-patched/     -lexternal-library-qpbo -DWITH_QPBO
endif
ifdef WITH_TRWS
  EXTLIBS += -I$(OPENGM_EXTSRC)/TRWS-v1.3.src-patched/     -lexternal-library-trws -DWITH_TRWS
endif
ifdef WITH_AD3
  EXTLIBS += -I$(OPENGM_EXTSRC)/AD3-patched/               -lexternal-library-ad3    -DWITH_AD3
endif
ifdef WITH_MRF
  EXTLIBS += -I$(OPENGM_EXTSRC)/MRF-v2.1.src-patched/      -lexternal-library-mrf -DMRFCOSTVALUE=double -DMRFENERGYVALUE=double -DMRFLABELVALUE=int  -DWITH_MRF
endif



all:  gridcv

gridcv: grid_potts_opencv.cxx
	g++ -o grid_potts_opencv grid_potts_opencv.cxx \
	-L/opt/local/lib -I/usr/local/include -O3 -DNDEBUG \
	`pkg-config --cflags opencv eigen3` `pkg-config --libs opencv eigen3` \
	-L$(OPENGM_EXTLIB) $(EXTLIBS)
