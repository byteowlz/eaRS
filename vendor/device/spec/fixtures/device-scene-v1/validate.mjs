// Run with bun; JSON Schema shape check uses an explicit Ajv installation supplied by the caller.
// This is a reference validator for fixture evidence, not the native runtime implementation.
import {readFile} from "node:fs/promises";
import {resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
const root=dirname(fileURLToPath(import.meta.url));
const ajvPath=process.env.DEVICE_SCENE_AJV;
if(!ajvPath)throw Error("Set DEVICE_SCENE_AJV to the installed ajv/dist/ajv.js path (Ajv 8).");
const {default:Ajv}=await import(ajvPath);
const schema=JSON.parse(await readFile(resolve(root,"../../device-scene-v1.schema.json"),"utf8"));
const validate=new Ajv({strict:true,allowUnionTypes:true,allErrors:true}).compile(schema);
const assert=(x,m)=>{if(!x)throw Error(m);};
function semantic(x){
  const ids=new Set(),controls=[];
  const unique=id=>{assert(!ids.has(id),"duplicate id");ids.add(id);};
  for(const d of x.scene.devices){unique(d.id);assert(d.chassis.front_thickness<d.chassis.depth,"front thickness");assert(x.scene.materials[d.chassis.material],"material reference");
    const panels=new Map(d.panels.map(p=>[p.id,p]));
    for(const p of d.panels){unique(p.id);assert(x.scene.materials[p.material],"material reference");}
    for(const c of d.controls){unique(c.id);controls.push(c);const p=panels.get(c.module_id);assert(p,"module reference");assert(x.scene.materials[c.material],"material reference");
      for(const [axis,size] of [["x","width"],["y","height"]])assert(Math.abs(c.rect[axis]-p.rect[axis])+c.rect[size]/2<=p.rect[size]/2+1e-6,"control bounds");
      const s=x.state.controls[c.id];assert(s,"missing state");const v=s.value;
      if(["knob","fader","meter"].includes(c.kind)){assert(c.range&&c.range.min<c.range.max,"range order");assert(typeof v==="number"&&v>=c.range.min&&v<=c.range.max,"numeric value");}
      if(c.kind==="select")assert(c.options?.some(o=>o.id===v),"option value");
      if(["text","shortcut","readout"].includes(c.kind))assert(typeof v==="string","string value");
      if(["toggle","indicator"].includes(c.kind))assert(typeof v==="boolean","boolean value");
      if(s.selection)assert(typeof v==="string"&&s.selection.start<=s.selection.end&&s.selection.end<=v.length,"selection bounds");
      if(c.matrix)assert(c.matrix.cols*c.matrix.cell_width*c.matrix.pitch+2*c.matrix.padding_x<=c.rect.width+1e-6&&c.matrix.rows*c.matrix.cell_height*c.matrix.pitch+2*c.matrix.padding_y<=c.rect.height+1e-6,"matrix fit");
    }
  }
  assert(Object.keys(x.state.controls).length===controls.length,"unknown state");
  assert(controls.filter(c=>x.state.controls[c.id].focused).length<=1,"multiple focus");
}
let checks=0;
const fixtures=[];
for(const name of ["ears-input","ears-shortcuts","spqx-receiver"]){
  const x=JSON.parse(await readFile(resolve(root,name+".json"),"utf8"));assert(validate(x),JSON.stringify(validate.errors));semantic(x);fixtures.push(x);checks++;
}
const rejects=(name,mutate)=>{const x=structuredClone(fixtures[0]);mutate(x);let rejected=false;try{assert(validate(x),"schema");semantic(x);}catch{rejected=true;}assert(rejected,"accepted negative: "+name);checks++;};
rejects("unknown root",x=>x.command="write files");
rejects("schema version",x=>x.schema_version=2);
rejects("negative depth",x=>x.scene.devices[0].chassis.depth=-1);
rejects("pixel cap",x=>x.viewport.pixel_cap=1e9);
rejects("invalid color",x=>x.scene.materials.face.base="red");
rejects("duplicate ID",x=>x.scene.devices[0].controls[1].id=x.scene.devices[0].controls[0].id);
rejects("unresolved panel",x=>x.scene.devices[0].controls[0].module_id="missing");
rejects("unresolved material",x=>x.scene.devices[0].controls[0].material="missing");
rejects("control outside panel",x=>x.scene.devices[0].controls[0].rect.x=999);
rejects("missing state",x=>delete x.state.controls["ears.input.device"]);
rejects("unknown option",x=>x.state.controls["ears.input.device"].value="real-device");
rejects("meter outside range",x=>x.state.controls["ears.input.level"].value=3);
rejects("selection bounds",x=>x.state.controls["ears.shortcuts.toggle"].selection={start:0,end:999,composing:false});
const actionSchema={...schema,$ref:"#/$defs/action"};const actionValidate=new Ajv({strict:true,allowUnionTypes:true}).compile(actionSchema);
for(const phase of ["preview","commit","cancel"]){assert(actionValidate({schema_version:1,scene_id:"ears.settings",scene_revision:1,state_revision:1,control_id:"ears.input.device",interaction_id:"interaction.1",sequence:1,phase,value:phase==="cancel"?null:"default",source:"keyboard"}),"action");checks++;}
console.log(checks+" contract checks passed (shape + reference semantic subset; native runtime owes complete invariants)");
