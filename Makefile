OPENGM_EXTSRC = /Users/tamaki/dev/opengm/src/external
OPENGM_EXTLIB = /Users/tamaki/dev/opengm/build/src/external
LIBDAI_LIB    = /Users/tamaki/dev/libdai/lib
EXTLIBS =

WITH_LIBDAI = 1
WITH_MAXFLOW = 1
WITH_QPBO = 1
WITH_TRWS = 1
WITH_AD3 = 1
WITH_MRF =
WITH_FASTPD = 1
WITH_MPLP = 1
WITH_GCO = 1
WITH_SAMPLING = 1

ifdef WITH_LIBDAI
  EXTLIBS += -L$(LIBDAI_LIB) -ldai -lgmp -lgmpxx -DWITH_LIBDAI
endif
ifdef WITH_MAXFLOW
  EXTLIBS += -I$(OPENGM_EXTSRC)/MaxFlow-v3.02.src-patched/ -lexternal-library-maxflow -DWITH_MAXFLOW
endif
ifdef WITH_QPBO
  EXTLIBS += -I$(OPENGM_EXTSRC)/QPBO-v1.3.src-patched/     -lexternal-library-qpbo    -DWITH_QPBO
endif
ifdef WITH_TRWS
  EXTLIBS += -I$(OPENGM_EXTSRC)/TRWS-v1.3.src-patched/     -lexternal-library-trws    -DWITH_TRWS
endif
ifdef WITH_AD3
  EXTLIBS += -I$(OPENGM_EXTSRC)/AD3-patched/               -lexternal-library-ad3     -DWITH_AD3
endif
ifdef WITH_MRF
  EXTLIBS += -I$(OPENGM_EXTSRC)/MRF-v2.1.src-patched/      -lexternal-library-mrf     -DWITH_MRF -DMRFCOSTVALUE=double -DMRFENERGYVALUE=double -DMRFLABELVALUE=int
endif
ifdef WITH_FASTPD
  EXTLIBS += -I$(OPENGM_EXTSRC)/FastPD.src-patched/        -lexternal-library-fastpd  -DWITH_FASTPD -DFASTPDENERGYVALUE=double -DFASTPDLABELVALUE=size_t
endif
ifdef WITH_MPLP
  EXTLIBS += -I$(OPENGM_EXTSRC)/mplp_ver2.src-patched/     -lexternal-library-mplp    -DWITH_MPLP
endif
ifdef WITH_GCO
  EXTLIBS += -I$(OPENGM_EXTSRC)/GCO-v3.0.src-patched/      -lexternal-library-gco     -DWITH_GCO -DGCOENERGYVALUE=double -DGCOLABELVALUE=int
endif
ifdef WITH_SAMPLING
  EXTLIBS += -DWITH_SAMPLING
endif


OPTIONS = -L/opt/local/lib -I/usr/local/include \
			`pkg-config --cflags opencv eigen3` `pkg-config --libs opencv eigen3` \
			-L$(OPENGM_EXTLIB) $(EXTLIBS) $(CFLAGS)
ifdef DEBUG
  OPTIONS+= -O0 -g -DDEBUG
else
  OPTIONS+= -O3 -DNDEBUG
endif

all:  gridcv stereo

gridcv: grid_potts_opencv.cxx
	g++ -o grid_potts_opencv grid_potts_opencv.cxx \
	$(OPTIONS)

stereo: stereo.cxx
	g++ -o stereo stereo.cxx \
	 -lboost_program_options-mt \
	$(OPTIONS)

clean:
	rm grid_potts_opencv stereo
