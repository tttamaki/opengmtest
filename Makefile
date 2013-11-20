OPENGM_EXTSRC = /Users/tamaki/Dropbox/opengm/src/external
OPENGM_EXTLIB = /Users/tamaki/Dropbox/opengm/build/src/external
LIBDAI_LIB    = /Users/tamaki/Dropbox/libdai/lib

all:  gridcv

gridcv: grid_potts_opencv.cxx
	g++ -o grid_potts_opencv grid_potts_opencv.cxx \
    -L/opt/local/lib -I/usr/local/include -O3 -DNDEBUG \
    `pkg-config --cflags opencv eigen3` `pkg-config --libs opencv eigen3` \
    -L$(OPENGM_EXTLIB) \
    -I$(OPENGM_EXTSRC)/MaxFlow-v3.02.src-patched/ -lexternal-library-maxflow \
    -I$(OPENGM_EXTSRC)/QPBO-v1.3.src-patched/     -lexternal-library-qpbo  \
    -I$(OPENGM_EXTSRC)/TRWS-v1.3.src-patched/     -lexternal-library-trws  \
    -I$(OPENGM_EXTSRC)/AD3-patched/               -lexternal-library-ad3   \
    -I$(OPENGM_EXTSRC)/MRF-v2.1.src-patched/      -lexternal-library-mrf -DMRFCOSTVALUE=double -DMRFENERGYVALUE=double -DMRFLABELVALUE=int \
    -L$(LIBDAI_LIB) -ldai -lgmp -lgmpxx

