CC = nvcc
OBJDIR = ./obj

DEPSNAMES = \
nvixnu__array_utils \
nvixnu__populate_arrays_utils \
nvixnu__error_utils \
nvixnu__cuda_devices_props

CFLAGS = -g -G --compiler-options -Wall -lm

INCLUDES = $(patsubst %,-I ../../%, $(DEPSNAMES))

# List with all .cu files inside $(REPOSDIR)/<repoName>
CUFILES = $(foreach dep,$(DEPSNAMES), $(wildcard ../../$(dep)/*.cu))

# List with all .o paths
OBJS = $(patsubst %.cu,%.o,$(CUFILES))

# Compiled objects path
COMPILEDOBJS := $(patsubst %,$(OBJDIR)/%,$(notdir $(OBJS))) 

# Creates the obj dir, compiles each dependency and then the main app
all: objdir $(OBJS)
	nvcc ch11__merge_sort.cu -o merge_sort.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

# Creates the ./obj dir
objdir:
	mkdir	-p	$(OBJDIR)

# Compile a dependency
%.o: %.cu
	nvcc -c $< -o $(OBJDIR)/$(notdir $@) $(CFLAGS)

# Run the executable
run:
	./main

# Remove the generated artifacts
clean:
	rm -Rf $(OBJDIR)/*.o
	rm -Rf merge_sort.out

.PHONY: all clean app run objdir