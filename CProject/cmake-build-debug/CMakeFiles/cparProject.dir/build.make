# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jose/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/203.7148.70/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/jose/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/203.7148.70/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jose/Desktop/CPAR/CProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jose/Desktop/CPAR/CProject/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cparProject.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cparProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cparProject.dir/flags.make

CMakeFiles/cparProject.dir/main.cpp.o: CMakeFiles/cparProject.dir/flags.make
CMakeFiles/cparProject.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jose/Desktop/CPAR/CProject/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cparProject.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cparProject.dir/main.cpp.o -c /home/jose/Desktop/CPAR/CProject/main.cpp

CMakeFiles/cparProject.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cparProject.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jose/Desktop/CPAR/CProject/main.cpp > CMakeFiles/cparProject.dir/main.cpp.i

CMakeFiles/cparProject.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cparProject.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jose/Desktop/CPAR/CProject/main.cpp -o CMakeFiles/cparProject.dir/main.cpp.s

# Object files for target cparProject
cparProject_OBJECTS = \
"CMakeFiles/cparProject.dir/main.cpp.o"

# External object files for target cparProject
cparProject_EXTERNAL_OBJECTS =

cparProject: CMakeFiles/cparProject.dir/main.cpp.o
cparProject: CMakeFiles/cparProject.dir/build.make
cparProject: CMakeFiles/cparProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jose/Desktop/CPAR/CProject/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cparProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cparProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cparProject.dir/build: cparProject

.PHONY : CMakeFiles/cparProject.dir/build

CMakeFiles/cparProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cparProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cparProject.dir/clean

CMakeFiles/cparProject.dir/depend:
	cd /home/jose/Desktop/CPAR/CProject/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jose/Desktop/CPAR/CProject /home/jose/Desktop/CPAR/CProject /home/jose/Desktop/CPAR/CProject/cmake-build-debug /home/jose/Desktop/CPAR/CProject/cmake-build-debug /home/jose/Desktop/CPAR/CProject/cmake-build-debug/CMakeFiles/cparProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cparProject.dir/depend

