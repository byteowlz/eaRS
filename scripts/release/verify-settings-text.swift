// Local-only OCR assertion on an explicitly supplied synthetic Settings capture.
// Does not capture the desktop or emit arbitrary recognized/private text.
import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count==2,let image=NSImage(contentsOfFile:CommandLine.arguments[1]),let cg=image.cgImage(forProposedRect:nil,context:nil,hints:nil) else{exit(2)}
let request=VNRecognizeTextRequest();request.recognitionLevel = .accurate;request.usesLanguageCorrection=false
try VNImageRequestHandler(cgImage:cg,options:[:]).perform([request])
let seen=(request.results ?? []).compactMap {$0.topCandidates(1).first?.string.lowercased()}
let expected=["insert","copy","prompt","toggle","push to talk","hybrid","local"]
let missing=expected.filter {word in !seen.contains(where:{$0==word})}
let value:[String:Any]=["ok":missing.isEmpty,"expected_labels":expected.count,"missing":missing]
let data=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]);print(String(data:data,encoding:.utf8)!)
exit(missing.isEmpty ? 0:1)
