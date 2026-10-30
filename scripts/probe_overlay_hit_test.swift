// Native, synthetic-only routing proof. Never starts capture, hotkeys or an ASR session.
// Run on an unlocked macOS desktop with existing event-posting permission:
// swift scripts/probe_overlay_hit_test.swift /absolute/path/to/ears-ui [--click-through|--expanded|--word|--floating]
import AppKit
import CoreGraphics

if CommandLine.arguments.count < 2 || !CGPreflightPostEventAccess() {
    print("BLOCKED: supply an ears-ui candidate and existing event-posting permission")
    exit(2)
}
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let clickThrough = CommandLine.arguments.contains("--click-through")
let expanded = CommandLine.arguments.contains("--expanded")
let word = CommandLine.arguments.contains("--word")
let floating = CommandLine.arguments.contains("--floating")
let child = Process()
child.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
child.arguments = [clickThrough ? "--preview-synthetic" : "--preview-hit-test"]
if expanded { child.arguments!.append("--preview-synthetic-expanded") }
if word { child.arguments!.append("--preview-word") }
if floating { child.arguments!.append("--preview-floating") }
let sandbox = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
try FileManager.default.createDirectory(at: sandbox, withIntermediateDirectories: true)
var environment = ProcessInfo.processInfo.environment
environment["XDG_CONFIG_HOME"] = sandbox.appendingPathComponent("config").path
environment["XDG_DATA_HOME"] = sandbox.appendingPathComponent("data").path
environment["XDG_STATE_HOME"] = sandbox.appendingPathComponent("state").path
child.environment = environment
child.standardOutput = FileHandle.nullDevice
let childErrors = Pipe(); child.standardError = childErrors
try child.run()
let originalPointer = CGEvent(source:nil)!.location
var received: [NSPoint] = []
var underlay: NSWindow?
var initialWindowCount = 0

func windows() -> [[String: Any]] {
    let all = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
    return all.filter { ($0[kCGWindowOwnerPID as String] as? Int32) == child.processIdentifier }
}
func finish(_ code: Int32, _ message: String) -> Never {
    underlay?.orderOut(nil)
    if child.isRunning {
        child.terminate()
        DispatchQueue.global().asyncAfter(deadline:.now()+1) {
            if child.isRunning { kill(child.processIdentifier, SIGKILL) }
        }
        child.waitUntilExit()
    }
    CGEvent(mouseEventSource:nil, mouseType:.mouseMoved, mouseCursorPosition:originalPointer, mouseButton:.left)?.post(tap:.cghidEventTap)
    try? FileManager.default.removeItem(at:sandbox)
    print(message)
    if code != 0 { print(String(data:childErrors.fileHandleForReading.readDataToEndOfFile(),encoding:.utf8) ?? "") }
    exit(code)
}
final class Target: NSView {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func mouseDown(with event: NSEvent) { received.append(event.locationInWindow) }
}
func click(_ rect: CGRect, _ x: CGFloat, _ y: CGFloat) {
    let point = CGPoint(x:rect.minX+x,y:rect.minY+y)
    // No dwell after crossing the shape: alpha routing must not depend on polling.
    for type in [CGEventType.mouseMoved, .leftMouseDown, .leftMouseUp] {
        CGEvent(mouseEventSource:nil,mouseType:type,mouseCursorPosition:point,mouseButton:.left)!.post(tap:.cghidEventTap)
    }
}
DispatchQueue.main.asyncAfter(deadline:.now()+1.2) {
    let candidates = windows()
    guard let info = candidates.first(where: { (($0[kCGWindowBounds as String] as? [String: Any])?["Width"] as? CGFloat) == 640 }),
          let bounds = info[kCGWindowBounds as String] as? [String: Any],
          let rect = CGRect(dictionaryRepresentation:bounds as CFDictionary),
          let screen = NSScreen.screens.first else {
        finish(3,"FAIL: candidate did not create the synthetic overlay")
    }
    initialWindowCount = candidates.count
    let cocoaRect = NSRect(x:rect.minX,y:screen.frame.maxY-rect.maxY,width:rect.width,height:rect.height)
    let window = NSWindow(contentRect:cocoaRect,styleMask:.borderless,backing:.buffered,defer:false)
    window.title = "eaRS synthetic click underlay"
    let target = Target(frame:NSRect(origin:.zero,size:cocoaRect.size))
    target.wantsLayer = true; target.layer!.backgroundColor = NSColor.systemBlue.cgColor
    window.contentView = target
    window.level = .floating // Must stay BELOW the popup even after receiving a click.
    window.orderFrontRegardless(); underlay = window
    // Empty margin, below the black silhouette, and just outside its bottom curve.
    let outside: [(CGFloat,CGFloat)] = floating && word ? [(10,80),(320,60),(170.5,62.5)] : (expanded ? [(10,85),(320,85),(40.5,67.5)] : [(10,50),(320,50),(40.5,31.5)])
    let inside: (CGFloat,CGFloat) = floating && word ? (320,80) : (100,16)
    for (i,p) in outside.enumerated() {
        DispatchQueue.main.asyncAfter(deadline:.now()+0.25+Double(i)*0.2) { click(rect,p.0,p.1) }
    }
    DispatchQueue.main.asyncAfter(deadline:.now()+0.95) { click(rect,inside.0,inside.1) }
    DispatchQueue.main.asyncAfter(deadline:.now()+1.4) {
        let expected = clickThrough ? 4 : 3
        guard received.count == expected else {
            finish(4,"FAIL: underlay received \(received.count) clicks, expected \(expected)")
        }
        let expectedPoints = outside + (clickThrough ? [inside] : [])
        for (actual, expectedPoint) in zip(received, expectedPoints) {
            if abs(actual.x-expectedPoint.0) > 1 || abs(actual.y-(rect.height-expectedPoint.1)) > 1 {
                finish(7,"FAIL: underlay received a click at an unexpected coordinate")
            }
        }
        if clickThrough && windows().count != initialWindowCount {
            finish(8,"FAIL: click-through mode unexpectedly opened a window")
        }
        if !clickThrough && windows().count <= initialWindowCount {
            finish(5,"FAIL: black-area click did not open Settings")
        }
        finish(0,"PASS: \(expected) underlay clicks; \(clickThrough ? "explicit full click-through" : "transparent margins/corner pass; black-area click opens Settings")")
    }
}
DispatchQueue.main.asyncAfter(deadline:.now()+8) { finish(6,"FAIL: bounded native fixture timed out") }
app.run()
