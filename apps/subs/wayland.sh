#!/usr/bin/env bash

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --dev       Run in development mode (default)
    --build     Build the application
    --run-bin   Run the compiled binary
    -h, --help  Show this help message

EOF
    exit 0
}

check_dependencies() {
    if ! pacman -Q webkitgtk-6.0 &>/dev/null; then
        echo "Error: webkitgtk-6.0 is not installed"
        echo "Install it with: sudo pacman -S webkitgtk-6.0"
        exit 1
    fi

    missing_packages=()
    for pkg in gst-plugins-good gst-plugins-bad gst-plugin-pipewire; do
        if ! pacman -Q "$pkg" &>/dev/null; then
            missing_packages+=("$pkg")
        fi
    done

    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "Error: Missing GStreamer plugins required for media device access:"
        printf '  - %s\n' "${missing_packages[@]}"
        echo "Install them with: sudo pacman -S ${missing_packages[*]}"
        exit 1
    fi
}

setup_environment() {
    export WEBKIT_DISABLE_DMABUF_RENDERER=1
    export GDK_BACKEND=x11
    export DISPLAY=:0
    export PATH=/opt/cuda/bin:$PATH
    export CUDA_ROOT=/opt/cuda
    export CUDARC_CUDA_VERSION=12060
    export SENTENCEPIECE_SYS_USE_PKG_CONFIG=1
    export WEBKIT_DISABLE_COMPOSITING_MODE=0
    export WEBKIT_ENABLE_MEDIA_STREAM=1
}

MODE="dev"

if [ $# -eq 0 ]; then
    MODE="dev"
else
    case "$1" in
        --dev)
            MODE="dev"
            ;;
        --build)
            MODE="build"
            ;;
        --run-bin)
            MODE="run-bin"
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
fi

check_dependencies
setup_environment

case "$MODE" in
    dev)
        echo "Starting Tauri with Hyprland-compatible settings..."
        exec bun tauri:linux
        ;;
    build)
        echo "Building Tauri with Hyprland-compatible settings..."
        exec bun tauri:linux:build
        ;;
    run-bin)
        echo "Starting ears-subs with Hyprland-compatible settings..."
        exec ./src-tauri/target/release/ears-subs
        ;;
esac
