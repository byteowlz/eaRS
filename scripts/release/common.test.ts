import {test,expect} from "bun:test";
import {mkdtemp,writeFile,symlink,rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {options,required,safeOutput,files,versionAtMost,xml} from "./common";

test("strict options reject malformed and duplicate arguments",()=>{
 expect(options(["--out","/tmp/package"])["--out"]).toBe("/tmp/package");
 for(const args of [["out","value"],["--out"],["--out","--next"],["--out","a","--out","b"]])expect(()=>options(args)).toThrow();
 expect(()=>required({},"--source-commit")).toThrow();
});
test("dangerous output roots and XML injection are rejected/escaped",()=>{
 for(const path of ["/",process.cwd(),process.env.HOME!])expect(()=>safeOutput(path)).toThrow();
 expect(xml('<name & "value">')).toBe('&lt;name &amp; &quot;value&quot;&gt;');
});
test("macOS versions compare components rather than decimal numbers",()=>{
 expect(versionAtMost("13.10","13.3")).toBe(false);
 expect(versionAtMost("13.0","13.0.0")).toBe(true);
 expect(versionAtMost("12.6","13.0")).toBe(true);
 expect(()=>versionAtMost("unknown","13.0")).toThrow();
});
test("inventory rejects symlink escapes rather than packaging private files",async()=>{
 const root=await mkdtemp(join(tmpdir(),"ears-package-test-"));
 try {await writeFile(join(root,"file"),"fixture");expect(await files(root)).toEqual(["file"]);await symlink("/etc/passwd",join(root,"escape"));await expect(files(root)).rejects.toThrow("symbolic link");}
 finally{await rm(root,{recursive:true});}
});
