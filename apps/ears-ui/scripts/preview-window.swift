import CoreGraphics
import Foundation

let target = Int(CommandLine.arguments[1])!
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
for window in windows {
    if let pid = window[kCGWindowOwnerPID as String] as? Int, pid == target,
       let number = window[kCGWindowNumber as String] as? Int,
       let bounds = window[kCGWindowBounds as String] as? [String: Any],
       let width = bounds["Width"] as? Double, width > 100 {
        print(number)
    }
}
