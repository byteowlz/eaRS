// Opt-in caller-owned C API probe; not a production eaRS adapter.
#include "needle.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <typename T>
static std::vector<T> read_file(const std::string& path, size_t limit) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open input file");
    auto size = file.tellg();
    if (size <= 0 || static_cast<size_t>(size) > limit || size % sizeof(T))
        throw std::runtime_error("Invalid bounded input size");
    std::vector<T> data(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("Truncated input file");
    return data;
}

static std::vector<float> read_pcm(const std::string& path) {
    auto pcm = read_file<float>(path, 480000 * sizeof(float));
    for (float value : pcm)
        if (!std::isfinite(value) || value < -1 || value > 1)
            throw std::runtime_error("PCM must be finite mono 16kHz float32 in [-1,1]");
    return pcm;
}

static std::string transcribe(const std::vector<float>& pcm, const char* language) {
    std::vector<char> output(65536, 0);
    if (needle_transcribe(pcm.data(), static_cast<int>(pcm.size()), language, nullptr,
                          1, output.data(), static_cast<int>(output.size())) < 0)
        throw std::runtime_error(needle_last_error());
    auto end = std::find(output.begin(), output.end(), '\0');
    if (end == output.end()) throw std::runtime_error("Native output is not terminated");
    return std::string(output.begin(), end);
}

static bool valid_language(const std::string& language) {
    return language == "auto" || language == "en" || language == "de" || language == "fr" ||
           language == "es" || language == "it" || language == "nl" || language == "pl";
}

static void run(const char* model_path, const char* manifest_path, const std::string& language) {
    if (!valid_language(language)) throw std::runtime_error("Unsupported explicit language");
    std::ifstream manifest(manifest_path);
    std::vector<std::string> paths;
    std::string path;
    while (std::getline(manifest, path)) {
        if (!path.empty()) paths.push_back(path);
        if (paths.size() > 64) throw std::runtime_error("At most 64 clips per spike batch");
    }
    if (paths.empty()) throw std::runtime_error("Empty/missing manifest");
    // Keep model bytes alive for the process-global, non-thread-safe native engine.
    auto model = read_file<unsigned char>(model_path, 32 * 1024 * 1024);
    auto start = Clock::now();
    if (needle_load(model.data(), model.size()) < 0) throw std::runtime_error(needle_last_error());
    double load_ms = elapsed_ms(start);
    if (needle_models() != NEEDLE_SPEECH) throw std::runtime_error("Expected speech model only");
    auto warm = read_pcm(paths.front());
    const char* hint = language == "auto" ? nullptr : language.c_str();
    struct SmallOutput { char before[8]; char text[2]; char after[8]; } small;
    std::fill(std::begin(small.before), std::end(small.before), 'Z');
    std::fill(std::begin(small.after), std::end(small.after), 'Z');
    std::fill(std::begin(small.text), std::end(small.text), '\0');
    int overflow = needle_transcribe(warm.data(), static_cast<int>(warm.size()), hint, nullptr, 1, small.text, 2);
    bool intact = std::all_of(std::begin(small.before), std::end(small.before), [](char x) { return x == 'Z'; }) &&
                  std::all_of(std::begin(small.after), std::end(small.after), [](char x) { return x == 'Z'; });
    if (!intact) throw std::runtime_error("Native small output buffer canary changed");
    start = Clock::now();
    (void)transcribe(warm, hint); // discarded warm-up after small-buffer probe
    double warm_ms = elapsed_ms(start);
    std::cout << "{\"kind\":\"meta\",\"load_ms\":" << load_ms
              << ",\"warmup_ms\":" << warm_ms << ",\"small_buffer_return\":" << overflow
              << ",\"small_buffer_canary_intact\":true,\"small_buffer_first_byte\":" << int(small.text[0]) << "}\n";
    for (size_t i = 0; i < paths.size(); ++i) {
        auto pcm = read_pcm(paths[i]);
        start = Clock::now();
        auto result = transcribe(pcm, hint);
        double proc_ms = elapsed_ms(start);
        std::cout << "{\"kind\":\"clip\",\"index\":" << i << ",\"proc_ms\":" << proc_ms
                  << ",\"audio_sec\":" << pcm.size() / 16000.0 << ",\"result\":" << result << "}\n";
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: native-bench MODEL.cact PCM_MANIFEST auto|LANGUAGE\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "Whistle spike: " << error.what() << '\n';
        return 1;
    }
}
