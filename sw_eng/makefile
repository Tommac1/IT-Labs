CC=g++

OBJDIR=.debug

TARGET=main
SRC=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJ=$(addprefix $(OBJDIR)/, $(SRC:.cpp=.o))


JSON_INCLUDE_PATH=/home/anon/Repositories/json/include
NLOHMANN_PATH=/home/anon/Repositories/json/include/nlohmann

CPP_FLAGS=-Wall -std=c++17 -pedantic -g -pthread
INCLUDE_PATHS=$(addprefix -I, \
			$(JSON_INCLUDE_PATH) $(NLOHMANN_PATH))

$(TARGET): $(OBJDIR) $(OBJ) $(HEADERS)
	$(CC) -o $@ $(OBJ) $(CPP_FLAGS) $(INCLUDE_PAJTHS)

$(OBJDIR)/%.o: %.cpp
	$(CC) -c -o $@ $< $(CPP_FLAGS) $(INCLUDE_PATHS)

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	rm -rf $(OBJ) $(TARGET) $(OBJDIR)
