// Local-only OCR assertion on an explicitly supplied synthetic Settings capture.
// Does not capture the desktop or emit arbitrary recognized/private text.
import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count==2,let image=NSImage(contentsOfFile:CommandLine.arguments[1]),let cg=image.cgImage(forProposedRect:nil,context:nil,hints:nil) else{exit(2)}
guard cg.width==780 && cg.height==592 else {
    fputs("Expected the default 780x592, 1x synthetic macOS Settings crop\n",stderr);exit(2)
}
let request=VNRecognizeTextRequest();request.recognitionLevel = .accurate;request.usesLanguageCorrection=false;request.recognitionLanguages=["en-US"]
// Default-size synthetic fixture's control column only; don't count hint text.
request.regionOfInterest=CGRect(x:Double(cg.width-320)/Double(cg.width),y:0.08,width:300.0/Double(cg.width),height:0.55)
try VNImageRequestHandler(cgImage:cg,options:[:]).perform([request])
// Vision reports Latin Copy as visually identical Cyrillic Соpу in this fixture.
// Normalize only these three glyph confusions for VISUAL testing, never for
// identifiers, commands or authority decisions. Truncated prefixes still fail.
let glyphs:[Character:Character]=["с":"c","о":"o","у":"y"]
let seen=(request.results ?? []).compactMap {$0.topCandidates(1).first?.string.lowercased()}.map {String($0.map {glyphs[$0] ?? $0})}
let expected=["insert","copy","prompt","toggle","push to talk","hybrid","local"]
// Vision may group adjacent choice cells into one observation. Require whole
// labels, not a particular observation grouping; prefixes such as "Cop" fail.
let missing=expected.filter {word in
    let pattern="(?<![a-z])"+NSRegularExpression.escapedPattern(for:word)+"(?![a-z])"
    return !seen.contains(where:{$0.range(of:pattern,options:.regularExpression) != nil})
}
let value:[String:Any]=["ok":missing.isEmpty,"expected_labels":expected.count,"missing":missing]
let data=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]);print(String(data:data,encoding:.utf8)!)
exit(missing.isEmpty ? 0:1)
