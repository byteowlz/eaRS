// Native Settings qualification for an already-running synthetic preview.
// No input injection, permission prompts, capture or real configuration access.
// swift scripts/probe_settings_buttons.swift PID [report|zoom|minimize|restore|close|resize W H|move X Y]
import ApplicationServices
import Foundation

guard CommandLine.arguments.count >= 2, let pid = pid_t(CommandLine.arguments[1]), pid > 0,
      AXIsProcessTrusted() else { fputs("An owned preview PID and existing Accessibility permission are required.\n", stderr); exit(2) }
func value(_ e: AXUIElement, _ key: String) -> CFTypeRef? {
    var v: CFTypeRef?
    return AXUIElementCopyAttributeValue(e, key as CFString, &v) == .success ? v : nil
}
func bounds(_ e: AXUIElement) -> [Double] {
    var p = CGPoint.zero; var s = CGSize.zero
    if let v = value(e, kAXPositionAttribute), CFGetTypeID(v) == AXValueGetTypeID() { AXValueGetValue(v as! AXValue, .cgPoint, &p) }
    if let v = value(e, kAXSizeAttribute), CFGetTypeID(v) == AXValueGetTypeID() { AXValueGetValue(v as! AXValue, .cgSize, &s) }
    return [p.x, p.y, s.width, s.height]
}
func check(_ error: AXError) { guard error == .success else { fputs("AX error \(error.rawValue)\n", stderr); exit(1) } }
let app = AXUIElementCreateApplication(pid)
func windows() -> [AXUIElement] { value(app, kAXWindowsAttribute) as? [AXUIElement] ?? [] }
guard let window = windows().first else { fputs("No preview window\n", stderr); exit(1) }
let action = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "report"
if action != "report" {
    let process = Process()
    process.executableURL = URL(fileURLWithPath:"/bin/ps")
    process.arguments = ["-p", String(pid), "-o", "command="]
    let pipe = Pipe()
    process.standardOutput = pipe
    try process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let args = String(data:data, encoding:.utf8)?.split(whereSeparator: { $0.isWhitespace }) ?? []
    guard process.terminationStatus == 0, args.contains("--preview-settings-device") else {
        fputs("Mutation is restricted to an owned --preview-settings-device process.\n", stderr)
        exit(2)
    }
}
switch action {
case "report": break
case "zoom", "minimize", "close":
    let key = action == "close" ? kAXCloseButtonAttribute : action == "zoom" ? kAXZoomButtonAttribute : kAXMinimizeButtonAttribute
    guard let b = value(window, key), CFGetTypeID(b) == AXUIElementGetTypeID() else { exit(1) }
    check(AXUIElementPerformAction(b as! AXUIElement, kAXPressAction as CFString))
case "restore": check(AXUIElementSetAttributeValue(window, kAXMinimizedAttribute as CFString, kCFBooleanFalse))
case "resize", "move":
    guard CommandLine.arguments.count == 5, let a = Double(CommandLine.arguments[3]), let b = Double(CommandLine.arguments[4]), a.isFinite, b.isFinite else { exit(2) }
    if action == "resize" {
        var size = CGSize(width:a, height:b)
        check(AXUIElementSetAttributeValue(window, kAXSizeAttribute as CFString, AXValueCreate(.cgSize, &size)!))
    } else {
        var p = CGPoint(x:a, y:b)
        check(AXUIElementSetAttributeValue(window, kAXPositionAttribute as CFString, AXValueCreate(.cgPoint, &p)!))
    }
default: exit(2)
}
RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.5))
let result = windows().map { w -> [String:Any] in
    var buttons: [String:Any] = [:]
    let r = bounds(w)
    for key in [kAXCloseButtonAttribute, kAXMinimizeButtonAttribute, kAXZoomButtonAttribute] {
        guard let button = value(w, key), CFGetTypeID(button) == AXUIElementGetTypeID() else { continue }
        let element = button as! AXUIElement
        let b = bounds(element)
        buttons[key] = ["bounds":b, "enabled":value(element, kAXEnabledAttribute) as? Bool ?? false,
                       "inside_window":b[0] >= r[0] && b[1] >= r[1] && b[0]+b[2] <= r[0]+r[2] && b[1]+b[3] <= r[1]+r[3]]
    }
    return ["bounds":r, "minimized":value(w, kAXMinimizedAttribute) as? Bool ?? false, "buttons":buttons]
}
let out: [String:Any] = ["action":action, "windows":result]
print(String(data:try! JSONSerialization.data(withJSONObject:out,options:[.sortedKeys]),encoding:.utf8)!)
