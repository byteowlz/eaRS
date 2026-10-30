// Synthetic native layout/placement proof. No pixels, microphone, ASR or real config.
import AppKit
import CoreGraphics

if CommandLine.arguments.count < 3 || !CGPreflightPostEventAccess() { print("BLOCKED: candidate, mode and existing event-posting access required"); exit(2) }
let mode=CommandLine.arguments[2]
let app=NSApplication.shared
app.setActivationPolicy(.accessory)
let originalPointer=CGEvent(source:nil)!.location
let originalFront=NSWorkspace.shared.frontmostApplication?.processIdentifier
let sandbox=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
try FileManager.default.createDirectory(at:sandbox,withIntermediateDirectories:true)
let child=Process()
child.executableURL=URL(fileURLWithPath:CommandLine.arguments[1])
child.arguments=["--preview-layout-report"]
switch mode {
case "word": child.arguments! += ["--preview-word","--preview-floating","--preview-departure"]
case "black": child.arguments! += ["--preview-word","--preview-floating","--preview-black"]
case "compact": child.arguments! += ["--preview-word","--preview-floating","--preview-black","--preview-compact"]
case "pointer", "pointer-cross-display": child.arguments! += ["--preview-word","--preview-floating","--preview-cursor"]
case "settings": child.arguments! += ["--preview-settings"]
case "minimum": child.arguments! += ["--preview-settings-minimum"]
case "light": child.arguments! += ["--preview-settings-light"]
default: print("Unknown mode");exit(2)
}
var environment=ProcessInfo.processInfo.environment
for key in ["CONFIG","DATA","STATE"] { environment["XDG_\(key)_HOME"]=sandbox.appendingPathComponent(key.lowercased()).path }
child.environment=environment
child.standardOutput=FileHandle.nullDevice
let pipe=Pipe();child.standardError=pipe
let lock=NSLock();var output=Data()
pipe.fileHandleForReading.readabilityHandler={ handle in let data=handle.availableData;lock.lock();output.append(data);lock.unlock() }
func reports()->[[String:Any]] {
    lock.lock();let text=String(data:output,encoding:.utf8) ?? "";lock.unlock()
    return text.components(separatedBy:"\n").compactMap { line in
        guard line.hasPrefix("ears-preview-layout: "),let data=line.dropFirst(21).data(using:.utf8) else {return nil}
        return (try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
    }
}
func latest(_ tag:String)->[String:Any]? {reports().last { ($0["tag"] as? String)==tag }}
func rectangles(_ report:[String:Any]?)->[[Double]] {report?["bounds"] as? [[Double]] ?? []}
func windows()->[[String:Any]] {
    let all=CGWindowListCopyWindowInfo(.optionOnScreenOnly,kCGNullWindowID) as? [[String:Any]] ?? []
    return all.filter { ($0[kCGWindowOwnerPID as String] as? Int32)==child.processIdentifier }
}
func rect(_ width:Double)->CGRect? {
    for info in windows() {
        if let bounds=info[kCGWindowBounds as String] as? [String:Any],let r=CGRect(dictionaryRepresentation:bounds as CFDictionary),abs(r.width-width)<2 {return r}
    };return nil
}
var ownedPointer:CGPoint?
func move(_ p:CGPoint) {ownedPointer=p;CGEvent(mouseEventSource:nil,mouseType:.mouseMoved,mouseCursorPosition:p,mouseButton:.left)!.post(tap:.cghidEventTap)}
func pointerMatches(_ p:CGPoint)->Bool {guard let actual=CGEvent(source:nil)?.location else{return false};return abs(actual.x-p.x)<2 && abs(actual.y-p.y)<2}
func requirePointer(_ p:CGPoint) {if !pointerMatches(p){finish(2,"BLOCKED: pointer changed during fixture; do not fight live desktop input")}}
func click(_ p:CGPoint) {
    ownedPointer=p
    for type in [CGEventType.mouseMoved,.leftMouseDown,.leftMouseUp] { CGEvent(mouseEventSource:nil,mouseType:type,mouseCursorPosition:p,mouseButton:.left)!.post(tap:.cghidEventTap) }
}
func key(_ code:CGKeyCode) {guard NSWorkspace.shared.frontmostApplication?.processIdentifier==child.processIdentifier else{finish(2,"BLOCKED: desktop focus changed during fixture")};for down in [true,false] { CGEvent(keyboardEventSource:nil,virtualKey:code,keyDown:down)!.post(tap:.cghidEventTap) }}
func finish(_ code:Int32,_ message:String)->Never {
    let sizes=windows().compactMap { $0[kCGWindowBounds as String] }
    let restoreFocus=NSWorkspace.shared.frontmostApplication?.processIdentifier==child.processIdentifier
    let restorePointer=ownedPointer.map(pointerMatches) ?? false
    if child.isRunning {child.terminate();child.waitUntilExit()}
    if restorePointer {move(originalPointer)}
    if restoreFocus,let pid=originalFront {NSRunningApplication(processIdentifier:pid)?.activate(options:[.activateIgnoringOtherApps])}
    try? FileManager.default.removeItem(at:sandbox)
    print(message)
    if code != 0 {print("owned window sizes: \(sizes)");lock.lock();let errors=String(data:output,encoding:.utf8) ?? "";lock.unlock();for line in errors.components(separatedBy:"\n") where !line.hasPrefix("ears-preview-layout: ") && !line.isEmpty {print(line)};for item in reports().suffix(12) {if let data=try? JSONSerialization.data(withJSONObject:item,options:[.sortedKeys]),let text=String(data:data,encoding:.utf8){print(text)}}}
    exit(code)
}
func verifySettings() {
    let tabs=rectangles(latest("settings-tabs")),root=rectangles(latest("settings-root"))
    guard tabs.count==4,root.count==2 else {finish(3,"FAIL: missing native settings geometry")}
    let width=tabs[0][2]
    guard tabs.allSatisfy({abs($0[2]-width)<1}),tabs.last![0]+width-tabs[0][0]>root[0][2]-60 else {finish(4,"FAIL: tabs do not share the header width")}
    guard root[1][1]>=root[0][1]+root[0][3]-1 else {finish(5,"FAIL: pane overlaps persistent header")}
    let viewport=(latest("settings-root")?["viewport"] as? [Double]) ?? []
    guard viewport.count==2,let w=rect(viewport[0]) else {finish(6,"FAIL: settings native window missing")}
    let chrome=w.height-viewport[1]
    let target=tabs[2]
    click(CGPoint(x:w.minX+target[0]+target[2]/2,y:w.minY+chrome+target[1]+target[3]/2))
    DispatchQueue.main.asyncAfter(deadline:.now()+0.4) {
        guard (latest("settings-tabs")?["state"] as? Int)==2 else {finish(7,"FAIL: Dictionary tab click did not select")}
        let buttons=rectangles(latest("dictionary-action"))
        guard let button=buttons.first,button[2]<200 else {finish(8,"FAIL: Add entry is not compact")}
        key(124)
        DispatchQueue.main.asyncAfter(deadline:.now()+0.3) {
            guard (latest("settings-tabs")?["state"] as? Int)==3 else {finish(9,"FAIL: right-arrow navigation")}
            key(115)
            DispatchQueue.main.asyncAfter(deadline:.now()+0.3) {
                guard (latest("settings-tabs")?["state"] as? Int)==0 else {finish(10,"FAIL: Home navigation")}
                finish(0,"PASS: \(mode) equal tabs/persistent header, compact action, native click/arrow/Home")
            }
        }
    }
}
let screen=CGDisplayBounds(CGMainDisplayID())
let first=CGPoint(x:screen.minX+screen.width/3,y:screen.minY+screen.height/3)
let second=CGPoint(x:first.x+140,y:first.y+90)
var displays=[CGDirectDisplayID](repeating:0,count:16);var count:UInt32=0
CGGetActiveDisplayList(16,&displays,&count)
if mode=="pointer-cross-display" && count<2 {print("SKIP: native cross-display needs two active displays");exit(0)}
if mode.hasPrefix("pointer") {move(first)}
try child.run()
DispatchQueue.main.asyncAfter(deadline:.now()+1.5) {
    if ["settings","minimum","light"].contains(mode){verifySettings();return}
    if mode.hasPrefix("pointer") {
        requirePointer(first)
        guard let before=rect(370) else {finish(11,"FAIL: pointer preview missing")}
        let other=displays.prefix(Int(count)).first { $0 != CGMainDisplayID() }
        let destination=mode=="pointer-cross-display" ? CGDisplayBounds(other!):screen
        let target=mode=="pointer-cross-display" ? CGPoint(x:destination.midX,y:destination.midY):second
        move(target)
        DispatchQueue.main.asyncAfter(deadline:.now()+0.6) {
            requirePointer(target)
            guard let after=rect(370) else {finish(12,"FAIL: native pointer preview disappeared")}
            if mode=="pointer-cross-display" {
                guard destination.contains(after.origin),after.maxX<=destination.maxX+1,after.maxY<=destination.maxY+1 else {finish(12,"FAIL: popup did not change displays")}
            }else if abs((after.minX-before.minX)-140)>=2||abs((after.minY-before.minY)-90)>=2 {finish(12,"FAIL: native bubble did not follow pointer")}
            finish(0,"PASS: \(mode) follows without losing visibility")
        };return
    }
    let body=rectangles(latest("overlay-body")),text=rectangles(latest("overlay-text"))
    guard let pill=body.last else {finish(13,"FAIL: native pill geometry missing")}
    if mode=="compact" {
        guard pill[2]<100,(latest("overlay-body")?["state"] as? Int)==0 else {finish(14,"FAIL: compact black option did not restore Classic")}
    }else {
        guard abs(pill[2]-300)<1,let label=text.first,abs((label[0]+label[2]/2)-(pill[0]+pill[2]/2))<1 else {finish(15,"FAIL: native word is not centered in fixed capsule")}
        let expected=mode=="black" ? 1:0
        guard (latest("overlay-body")?["state"] as? Int)==expected else {finish(16,"FAIL: resolved skin selection")}
    }
    finish(0,"PASS: \(mode) native word geometry and independent resolved surface")
}
DispatchQueue.main.asyncAfter(deadline:.now()+7){finish(17,"FAIL: bounded fixture timed out")}
app.run()
