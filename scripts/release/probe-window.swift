// Numeric window evidence for a caller-owned test process. No pixels or titles.
import CoreGraphics
import Foundation

guard CommandLine.arguments.count == 2, let pid=Int32(CommandLine.arguments[1]), pid>0 else {exit(2)}
let windows=(CGWindowListCopyWindowInfo(.optionOnScreenOnly,kCGNullWindowID) as? [[String:Any]] ?? []).filter {($0[kCGWindowOwnerPID as String] as? Int32)==pid}
let sizes=windows.compactMap {$0[kCGWindowBounds as String] as? [String:Double]}
let value:[String:Any]=["window_count":sizes.count,"bounds":sizes]
let data=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])
print(String(data:data,encoding:.utf8)!)
exit(sizes.isEmpty ? 1:0)
