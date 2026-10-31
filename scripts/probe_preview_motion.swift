// Numeric-only observation of an OWNED synthetic preview in an isolated desktop.
// Never posts input, captures pixels/titles/text, grants permissions or controls apps.
import Foundation
import CoreGraphics
import AppKit

guard CommandLine.arguments.count==3,let pid=Int32(CommandLine.arguments[1]),let seconds=Double(CommandLine.arguments[2]),seconds>=1,seconds<=15 else{exit(2)}
struct Sample {let t:Double;let x:Double;let y:Double;let pointer:CGPoint;let pointerBefore:CGPoint;let queryMs:Double}
func preview() -> CGRect? {
    let windows=CGWindowListCopyWindowInfo([.optionOnScreenOnly,.excludeDesktopElements],kCGNullWindowID) as? [[String:Any]] ?? []
    for window in windows where (window[kCGWindowOwnerPID as String] as? Int32)==pid {
        guard let data=window[kCGWindowBounds as String] as? [String:Double],let x=data["X"],let y=data["Y"],let width=data["Width"],let height=data["Height"] else{continue}
        let rect=CGRect(x:x,y:y,width:width,height:height)
        if rect.width>=350 && rect.width<=400 && rect.height>=140 && rect.height<=170 {return rect}
    }
    return nil
}
func percentile(_ values:[Double],_ fraction:Double)->Double? {
    let values=values.sorted();guard !values.isEmpty else{return nil}
    return values[min(values.count-1,Int(Double(values.count-1)*fraction))]
}
var samples:[Sample]=[]
let start=ProcessInfo.processInfo.systemUptime
while ProcessInfo.processInfo.systemUptime-start<seconds {
    let queryStart=ProcessInfo.processInfo.systemUptime
    if let before=CGEvent(source:nil),let rect=preview(),let event=CGEvent(source:nil) {
        samples.append(Sample(t:ProcessInfo.processInfo.systemUptime-start,x:rect.minX,y:rect.minY,pointer:event.location,pointerBefore:before.location,queryMs:(ProcessInfo.processInfo.systemUptime-queryStart)*1000))
    }
    Thread.sleep(forTimeInterval:0.004)
}
let changes=(1..<max(1,samples.count)).filter {samples[$0].pointer != samples[$0-1].pointer}
guard let first=changes.first,let last=changes.last,last>first else {
    print("{\"ok\":false,\"reason\":\"No synthetic pointer motion observed\"}");exit(1)
}
let active=Array(samples[first...last])
var moved:[Double]=[]
for i in 1..<active.count where abs(active[i].x-active[i-1].x)>0.1 || abs(active[i].y-active[i-1].y)>0.1 {moved.append(active[i].t)}
let gaps=zip(moved.dropFirst(),moved).map{($0-$1)*1000}
// Fixture trajectory stays away from edges; incumbent offsets are +24,+28.
let errors=active.map{hypot($0.x-($0.pointer.x+24),$0.y-($0.pointer.y+28))}
let beforeErrors=active.map{hypot($0.x-($0.pointerBefore.x+24),$0.y-($0.pointerBefore.y+28))}
let result:[String:Any]=["screen_maximum_hz":NSScreen.screens.first?.maximumFramesPerSecond ?? 0,"window_query_p95_ms":percentile(active.map{$0.queryMs},0.95) ?? -1,"pointer_offset_before_query_p95_px":percentile(beforeErrors,0.95) ?? -1,"ok":!gaps.isEmpty,"observer":"public CGWindow bounds; not physical compositor scanout","samples":active.count,"native_moves":moved.count,"motion_seconds":active.last!.t-active.first!.t,"move_gap_p50_ms":percentile(gaps,0.5) ?? -1,"move_gap_p95_ms":percentile(gaps,0.95) ?? -1,"move_gap_max_ms":gaps.max() ?? -1,"pointer_offset_error_p50_px":percentile(errors,0.5) ?? -1,"pointer_offset_error_p95_px":percentile(errors,0.95) ?? -1]
let data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);print(String(data:data,encoding:.utf8)!)
