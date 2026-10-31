import {test,expect} from "bun:test";
import {cadence} from "./check_preview_motion";
const base={ok:true,motion_seconds:4,native_moves:115,move_gap_p50_ms:33.53,move_gap_p95_ms:50.18};
test("30Hz movement fails a60Hz replay and display-paced movement meets fixed budgets",()=>{
 expect(cadence(base,60).pass).toBe(false);
 expect(cadence({...base,native_moves:225,move_gap_p50_ms:16.96,move_gap_p95_ms:22.93},60).pass).toBe(true);
 expect(cadence({...base,move_gap_p50_ms:10,move_gap_p95_ms:25.01},60).pass).toBe(false);
 expect(cadence({...base,move_gap_p50_ms:8.33,move_gap_p95_ms:12.5},120).pass).toBe(true);
 expect(cadence({...base,move_gap_p50_ms:8.33,move_gap_p95_ms:12.51},120).pass).toBe(false);
});
test("missing, nonfinite, short or empty observations never qualify",()=>{
 for(const changed of [{ok:false},{motion_seconds:1},{native_moves:0},{move_gap_p50_ms:NaN},{move_gap_p95_ms:Infinity}])expect(()=>cadence({...base,...changed},60)).toThrow();
 for(const hz of [0,29,241,NaN])expect(()=>cadence(base,hz)).toThrow();
});
