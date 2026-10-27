// Inspect only the synthetic settings fixture, never a real user's screen.
import AppKit
import Foundation
import Vision

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
try VNImageRequestHandler(url: url).perform([request])
let apply = (request.results ?? []).filter {
    $0.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines) == "Apply"
}
guard apply.count == 1, let box = apply.first?.boundingBox, box.maxX < 0.98 else {
    fputs("Synthetic settings Apply label is missing or clipped\n", stderr)
    exit(1)
}
print("synthetic settings: Apply label fully visible")
