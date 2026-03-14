# RDKS100 Drowning Detection Project Makefile
# 集成tools文件夹中的所有测试脚本到一个统一的Makefile

# 配置变量
BUILD_DIR ?= build
TEST_DIR = $(BUILD_DIR)/test
SRC_DIR = $(BUILD_DIR)/src
PROJECT_ROOT = $(shell pwd)
DEFAULT_USER = sunrise
DEFAULT_PATH = /home/$(DEFAULT_USER)/Desktop/RDKS100_Drowning

# 检测是否在默认路径下运行
ifeq ($(shell pwd),$(DEFAULT_PATH))
    IS_DEFAULT_PATH := true
else
    IS_DEFAULT_PATH := false
endif

# 编译目标
.PHONY: all build rebuild clean install

all: build

# 主要构建目标
build:
	@echo "🔨 正在构建项目..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make -j4
	@echo "✅ 构建完成！"

# 重新构建（清理后构建）
rebuild: clean build

# 清理构建文件
clean:
	@echo "🧹 正在清理构建文件..."
	@rm -rf $(BUILD_DIR)
	@echo "✅ 清理完成！"

# 安装/部署目标
install: build
	@echo "📦 正在安装..."
	@cd $(BUILD_DIR) && sudo make install
	@echo "✅ 安装完成！"

# 主程序运行目标
.PHONY: run run-main

# 运行主程序
run run-main: build
	@echo "🚀 正在启动主程序..."
	@cd $(SRC_DIR) && sudo ./RDKS100_Drowning

# 测试目标组
.PHONY: test-all test-basic test-advanced test-clean

# 运行所有测试
test-all: test-basic test-advanced
	@echo "✅ 所有测试完成！"

# 基础测试（核心功能）
test-basic: build
	@echo "🧪 正在运行基础测试..."
	@cd $(TEST_DIR) && \
	if [ -f test_Bytetrack ]; then ./test_Bytetrack && echo "✅ Bytetrack测试通过"; fi; \
	if [ -f test_Detection ]; then ./test_Detection && echo "✅ Detection测试通过"; fi; \
	if [ -f test_ImageSaver ]; then ./test_ImageSaver && echo "✅ ImageSaver测试通过"; fi; \
	if [ -f test_rtsp ]; then sudo ./test_rtsp && echo "✅ RTSP测试通过"; fi

# 高级测试（特定模型和功能）
test-advanced: build
	@echo "🧪 正在运行高级测试..."
	@cd $(TEST_DIR) && \
	if [ -f test_H264 ]; then ./test_H264 && echo "✅ H264测试通过"; fi; \
	if [ -f test_Pose ]; then ./test_Pose && echo "✅ Pose测试通过"; fi; \
	if [ -f test_seg ]; then ./test_seg && echo "✅ Seg测试通过"; fi; \
	if [ -f test_SAHI ]; then ./test_SAHI && echo "✅ SAHI测试通过"; fi; \
	if [ -f test_deeplabv3+ ]; then ./test_deeplabv3+ && echo "✅ DeepLabV3+测试通过"; fi; \
	if [ -f test_Drowning ]; then ./test_Drowning && echo "✅ Drowning测试通过"; fi; \
	if [ -f test_HIKSDK ]; then ./test_HIKSDK && echo "✅ HIKSDK测试通过"; fi; \
	if [ -f test_Cereal_SDK ]; then ./test_Cereal_SDK && echo "✅ Cereal_SDK测试通过"; fi; \
	if [ -f test_Polg ]; then ./test_Polg && echo "✅ Polg测试通过"; fi; \
	if [ -f test_pose_Base ]; then ./test_pose_Base && echo "✅ pose_Base测试通过"; fi; \
	if [ -f test_roughwaternet ]; then ./test_roughwaternet && echo "✅ roughwaternet测试通过"; fi; \
	if [ -f test_finewaternet ]; then ./test_finewaternet && echo "✅ finewaternet测试通过"; fi; \
	if [ -f test_mmpose ]; then ./test_mmpose && echo "✅ mmpose测试通过"; fi; \
	if [ -f test_patchcore ]; then ./test_patchcore && echo "✅ patchcore测试通过"; fi; \
	if [ -f test_patchcore_hpp ]; then ./test_patchcore_hpp && echo "✅ patchcore_hpp测试通过"; fi; \
	if [ -f test_swimmer ]; then ./test_swimmer && echo "✅ swimmer测试通过"; fi

# 清理测试输出
test-clean:
	@echo "🧹 正在清理测试输出..."
	@find . -name "*.log" -o -name "*.out" -o -name "test_*" | grep -v ".cc" | xargs rm -f 2>/dev/null || true
	@echo "✅ 测试输出清理完成！"

# 特定功能测试目标
.PHONY: test-bytetrack test-detection test-imagesaver test-rtsp test-h264
.PHONY: test-hikcamera test-pose test-seg test-sahi test-deeplab
.PHONY: test-drowning test-swimmer test-under-surface

# ByteTrack 跟踪测试
test-bytetrack: build
	@echo "🎯 正在测试ByteTrack..."
	@cd $(TEST_DIR) && ./test_Bytetrack

# 目标检测测试
test-detection: build
	@echo "🔍 正在测试目标检测..."
	@cd $(TEST_DIR) && ./test_Detection

# 图像保存测试
test-imagesaver: build
	@echo "💾 正在测试图像保存..."
	@cd $(TEST_DIR) && ./test_ImageSaver

# RTSP摄像头测试
test-rtsp: build
	@echo "📹 正在测试RTSP摄像头..."
	@cd $(TEST_DIR) && sudo ./test_rtsp

# H264编解码测试
test-h264: build
	@echo "🎬 正在测试H264编解码..."
	@cd $(TEST_DIR) && ./test_H264

# HIKSDK测试
test-hiksdk: build
	@echo "🏭 正在测试HIKSDK..."
	@cd $(TEST_DIR) && ./test_HIKSDK

# 海康摄像头测试
test-hikcamera: build
	@echo "🏭 正在测试海康摄像头..."
	@cd $(TEST_DIR) && ./test_HikCamera

# 海康显示测试
test-show-hik: build
	@echo "📺 正在测试海康显示..."
	@cd $(TEST_DIR) && ./test_Show_HIK

# 姿态估计测试
test-pose: build
	@echo "🧍 正在测试姿态估计..."
	@cd $(TEST_DIR) && ./test_Pose

# 姿态估计图像测试
test-pose-img: build
	@echo "🧍 正在测试姿态估计图像..."
	@cd $(TEST_DIR) && ./test_Pose_img

# 基础姿态测试
test-pose-base: build
	@echo "🧍 正在测试基础姿态..."
	@cd $(TEST_DIR) && ./test_pose_Base

# MMPose测试
test-mmpose: build
	@echo "🧍 正在测试MMPose..."
	@cd $(TEST_DIR) && ./test_mmpose

# 语义分割测试
test-seg: build
	@echo "🗺️ 正在测试语义分割..."
	@cd $(TEST_DIR) && ./test_seg

# SAHI小目标检测测试
test-sahi: build
	@echo "🔬 正在测试SAHI小目标检测..."
	@cd $(TEST_DIR) && ./test_SAHI

# DeepLabV3+测试
test-deeplab: build
	@echo "🌊 正在测试DeepLabV3+..."
	@cd $(TEST_DIR) && ./test_deeplabv3+

# Cereal SDK测试
test-cereal: build
	@echo "🥣 正在测试Cereal SDK..."
	@cd $(TEST_DIR) && ./test_Cereal_SDK

# Plog测试
test-plog: build
	@echo "📝 正在测试Plog日志..."
	@cd $(TEST_DIR) && ./test_Polg

# 粗糙水体网络测试
test-roughwaternet: build
	@echo "🌊 正在测试粗糙水体网络..."
	@cd $(TEST_DIR) && ./test_roughwaternet

# 精细水体网络测试
test-finewaternet: build
	@echo "💧 正在测试精细水体网络..."
	@cd $(TEST_DIR) && ./test_finewaternet

# PatchCore测试
test-patchcore: build
	@echo "🔧 正在测试PatchCore..."
	@cd $(TEST_DIR) && ./test_patchcore

# PatchCore HPP测试
test-patchcore-hpp: build
	@echo "🔧 正在测试PatchCore HPP..."
	@cd $(TEST_DIR) && ./test_patchcore_hpp

# 溺水检测测试
test-drowning: build
	@echo "⚠️ 正在测试溺水检测..."
	@cd $(TEST_DIR) && ./test_Drowning

# 游泳者检测测试
test-swimmer: build
	@echo "🏊 正在测试游泳者检测..."
	@cd $(TEST_DIR) && ./test_swimmer

# 水下/水面状态检测测试
test-under-surface: build
	@echo "🌊 正在测试水下水面状态检测..."
	@cd $(TEST_DIR) && ./test_UnderOrSurface

# 开发辅助目标
.PHONY: info help dev-setup debug

# 显示项目信息
info:
	@echo "📋 项目信息:"
	@echo "  项目名称: RDKS100 Drowning Detection"
	@echo "  项目路径: $(PROJECT_ROOT)"
	@echo "  构建目录: $(BUILD_DIR)"
	@echo "  默认路径: $(DEFAULT_PATH)"
	@echo "  是否在默认路径: $(IS_DEFAULT_PATH)"
	@echo ""
	@echo "📁 可用测试:"
	@ls -1 test/test_*.cc 2>/dev/null | sed 's/test\/test_//g' | sed 's/\.cc//g' | sed 's/^/  - /'

# 显示帮助信息
help:
	@echo "📚 RDKS100 Drowning Detection Makefile 帮助"
	@echo ""
	@echo "🎯 主要目标:"
	@echo "  make build          - 构建项目"
	@echo "  make rebuild        - 重新构建（清理后构建）"
	@echo "  make run            - 运行主程序"
	@echo "  make clean          - 清理构建文件"
	@echo ""
	@echo "🧪 测试目标:"
	@echo "  make test-all       - 运行所有测试"
	@echo "  make test-basic     - 运行基础测试"
	@echo "  make test-advanced  - 运行高级测试"
	@echo ""
	@echo "🔧 特定功能测试:"
	@echo "  make test-bytetrack     - ByteTrack跟踪测试"
	@echo "  make test-detection     - 目标检测测试"
	@echo "  make test-rtsp         - RTSP摄像头测试"
	@echo "  make test-pose         - 姿态估计测试"
	@echo "  make test-drowning     - 溺水检测测试"
	@echo "  make test-seg          - 语义分割测试"
	@echo "  make test-deeplab      - DeepLabV3+测试"
	@echo "  make test-sahi         - SAHI小目标检测测试"
	@echo "  make test-hiksdk       - HIKSDK测试"
	@echo "  make test-hikcamera    - 海康摄像头测试"
	@echo "  make test-show-hik     - 海康显示测试"
	@echo "  make test-pose-img     - 姿态估计图像测试"
	@echo "  make test-pose-base    - 基础姿态测试"
	@echo "  make test-mmpose       - MMPose测试"
	@echo "  make test-cereal       - Cereal SDK测试"
	@echo "  make test-plog         - Plog日志测试"
	@echo "  make test-roughwaternet - 粗糙水体网络测试"
	@echo "  make test-finewaternet - 精细水体网络测试"
	@echo "  make test-patchcore    - PatchCore测试"
	@echo "  make test-patchcore-hpp - PatchCore HPP测试"
	@echo ""
	@echo "ℹ️  辅助目标:"
	@echo "  make info           - 显示项目信息"
	@echo "  make help           - 显示此帮助信息"
	@echo "  make dev-setup      - 开发环境设置"

# 开发环境设置
dev-setup:
	@echo "⚙️  正在设置开发环境..."
	@echo "✅ 确保已安装以下依赖:"
	@echo "  - cmake >= 3.10"
	@echo "  - OpenCV"
	@echo "  - Qt5"
	@echo "  - Horizon BPU SDK"
	@echo "  - Eigen3"
	@echo "  - FFmpeg"
	@echo ""
	@echo "💡 使用 'make build' 开始构建项目"

# 调试信息
debug:
	@echo "🐛 调试信息:"
	@echo "  SHELL: $(SHELL)"
	@echo "  MAKE_VERSION: $(shell make --version | head -1)"
	@echo "  PROJECT_ROOT: $(PROJECT_ROOT)"
	@echo "  BUILD_DIR: $(BUILD_DIR)"
	@echo "  TEST_DIR: $(TEST_DIR)"
	@echo "  SRC_DIR: $(SRC_DIR)"
	@echo "  IS_DEFAULT_PATH: $(IS_DEFAULT_PATH)"
	@echo ""
	@echo "📊 构建状态:"
	@if [ -d "$(BUILD_DIR)" ]; then \
	    echo "  构建目录: 存在"; \
	    if [ -f "$(SRC_DIR)/RDKS100_Drowning" ]; then \
	        echo "  主程序: 已构建"; \
	    else \
	        echo "  主程序: 未构建"; \
	    fi; \
	else \
	    echo "  构建目录: 不存在"; \
	fi

# 默认目标
.DEFAULT_GOAL := help