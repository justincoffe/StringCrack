#---------------------------------------------------------------------
# V3 Goldilocks Makefile for VanitySearch (CUDA 12.8 / GCC 11)
#---------------------------------------------------------------------

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o Wildcard.o)

CXX        = g++-11
CUDA       = /usr/local/cuda
CXXCUDA    = g++-11
NVCC       = $(CUDA)/bin/nvcc

# CPU Optimization: -O2 and -fno-strict-aliasing are MANDATORY to protect 
# the legacy pointer punning in Int.cpp and SECP256K1.cpp from breaking.
# We also restore -mssse3 to protect the native SSE hash files.
ifdef debug
CXXFLAGS   = -g -Wno-write-strings -I. -I$(CUDA)/include
else
CXXFLAGS   = -O2 -march=native -fno-strict-aliasing -mssse3 -Wno-write-strings -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

# GPU Architecture Targets (Ada, Hopper, Blackwell)
GENCODE    = -gencode=arch=compute_89,code=sm_89 \
             -gencode=arch=compute_90,code=sm_90 \
             -gencode=arch=compute_120,code=sm_120

#--------------------------------------------------------------------

ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include $(GENCODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 -Xptxas -O3 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -I$(CUDA)/include $(GENCODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: VanitySearch

VanitySearch: $(OBJET)
	@echo "Linking V3 Goldilocks VanitySearch..."
	$(CXX) $(OBJET) $(LFLAGS) -o vanitysearch

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p hash

clean:
	@echo "Cleaning up..."
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
	@rm -f vanitysearch
