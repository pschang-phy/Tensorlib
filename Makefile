
CXX = g++

# Flags for optimized executables
CXXFLAGS = -I./lib -O2

# Flags for debugging
# CXXFLAGS = -std=gnu++11 -g

LDFLAGS =
LIBS = -lblas -llapacke


LIB_SRC = $(wildcard lib/*.cpp)
LIB_OBJ = $(notdir $(LIB_SRC:.cpp=.o))


.SUFFIXES: .o .cpp

itebd: itebd.o $(LIB_OBJ)
	$(CXX) $(LDFLAGS) $^ -o itebd $(LIBS)

itebd.o: itebd.cpp
	$(CXX) $(CXXFLAGS) -c itebd.cpp

%.o: lib/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(TARGETS) itebd
