# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/thomle/opt/anaconda3/lib/python3.9/site-packages/cmake/data/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Users/thomle/opt/anaconda3/lib/python3.9/site-packages/cmake/data/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling

# Include any dependencies generated for this target.
include CMakeFiles/GraphSampling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GraphSampling.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GraphSampling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GraphSampling.dir/flags.make

CMakeFiles/GraphSampling.dir/main.cpp.o: CMakeFiles/GraphSampling.dir/flags.make
CMakeFiles/GraphSampling.dir/main.cpp.o: main.cpp
CMakeFiles/GraphSampling.dir/main.cpp.o: CMakeFiles/GraphSampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GraphSampling.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GraphSampling.dir/main.cpp.o -MF CMakeFiles/GraphSampling.dir/main.cpp.o.d -o CMakeFiles/GraphSampling.dir/main.cpp.o -c /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/main.cpp

CMakeFiles/GraphSampling.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GraphSampling.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/main.cpp > CMakeFiles/GraphSampling.dir/main.cpp.i

CMakeFiles/GraphSampling.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GraphSampling.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/main.cpp -o CMakeFiles/GraphSampling.dir/main.cpp.s

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o: CMakeFiles/GraphSampling.dir/flags.make
CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o: cnpy/cnpy.cpp
CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o: CMakeFiles/GraphSampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o -MF CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.d -o CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o -c /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/cnpy/cnpy.cpp

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/cnpy/cnpy.cpp > CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/cnpy/cnpy.cpp -o CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s

# Object files for target GraphSampling
GraphSampling_OBJECTS = \
"CMakeFiles/GraphSampling.dir/main.cpp.o" \
"CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o"

# External object files for target GraphSampling
GraphSampling_EXTERNAL_OBJECTS =

GraphSampling: CMakeFiles/GraphSampling.dir/main.cpp.o
GraphSampling: CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o
GraphSampling: CMakeFiles/GraphSampling.dir/build.make
GraphSampling: /usr/local/lib/libopencv_gapi.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_stitching.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_alphamat.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_aruco.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_barcode.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_bgsegm.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_bioinspired.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_ccalib.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_dnn_objdetect.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_dnn_superres.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_dpm.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_face.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_freetype.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_fuzzy.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_hfs.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_img_hash.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_intensity_transform.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_line_descriptor.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_mcc.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_quality.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_rapid.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_reg.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_rgbd.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_saliency.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_sfm.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_stereo.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_structured_light.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_superres.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_surface_matching.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_tracking.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_videostab.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_viz.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_wechat_qrcode.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_xfeatures2d.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_xobjdetect.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_xphoto.4.6.0.dylib
GraphSampling: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX12.1.sdk/usr/lib/libz.tbd
GraphSampling: /usr/local/lib/libopencv_shape.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_highgui.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_datasets.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_plot.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_text.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_ml.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_phase_unwrapping.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_optflow.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_ximgproc.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_video.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_videoio.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_imgcodecs.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_objdetect.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_calib3d.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_dnn.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_features2d.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_flann.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_photo.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_imgproc.4.6.0.dylib
GraphSampling: /usr/local/lib/libopencv_core.4.6.0.dylib
GraphSampling: CMakeFiles/GraphSampling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GraphSampling"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GraphSampling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GraphSampling.dir/build: GraphSampling
.PHONY : CMakeFiles/GraphSampling.dir/build

CMakeFiles/GraphSampling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GraphSampling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GraphSampling.dir/clean

CMakeFiles/GraphSampling.dir/depend:
	cd /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling /Users/thomle/Research/ASIA_Lab/3D_printing_medical_AI/Implements/github_collect/MeshConvolution/code/GraphSampling/CMakeFiles/GraphSampling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GraphSampling.dir/depend
