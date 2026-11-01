"""Render digital twin images using native USD XformOp stacking for perfectly centered defects."""
import argparse, json, os, subprocess, sys, traceback, time, random
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"])
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"))
ap.add_argument("--damaged-vial", default=os.path.join(HERE, "damaged_vial.usdc"))
ap.add_argument("--frame", type=int, default=1)
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32)
ap.add_argument("--kit-arg", action="append", default=[])
a, _ = ap.parse_known_args()

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = Path(HERE) / stage_path
stage_path = str(stage_path.resolve())

Path(a.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
a.out = str(Path(a.out).expanduser().resolve())

rw, rh = int(cam["render_px"][0]), int(cam["render_px"][1])

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    import omni.usd, omni.timeline
    from pxr import Gf, UsdGeom, UsdLux
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(20):
        kit.update()
        
    stage = ctx.get_stage()
    prim_path = cam["prim_in_station_file"] if stage.GetPrimAtPath(cam["prim_in_station_file"]).IsValid() else cam["prim_in_csi_line"]
    prim = stage.GetPrimAtPath(prim_path)

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    tl.pause()
    kit.update()

    viewport = get_active_viewport()
    viewport.camera_path = prim_path
    kit.update()

    clip = UsdGeom.Camera(prim).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    prims = [p for p in stage.Traverse()]
    caps = [p for p in prims if "cap" in p.GetName().lower() and "cam" not in p.GetName().lower() and p.IsA(UsdGeom.Imageable)]
    vials = [p for p in prims if "vial" in p.GetName().lower() and "cap" not in p.GetName().lower() and "cam" not in p.GetName().lower() and p.IsA(UsdGeom.Imageable)]
    
    print(f"\n--- FOUND {len(caps)} CAPS AND {len(vials)} VIALS ---", flush=True)

    light_path = prim_path + "/CameraFlashlight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateIntensityAttr(85000)
        light.CreateRadiusAttr(5.0)
        light.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

    pockets = {}
    for c in caps:
        parent = c.GetParent().GetPath().pathString
        if parent not in pockets: pockets[parent] = {'cap': None, 'vial': None}
        pockets[parent]['cap'] = c
    for v in vials:
        parent = v.GetParent().GetPath().pathString
        if parent not in pockets: pockets[parent] = {'cap': None, 'vial': None}
        pockets[parent]['vial'] = v

    # Cache exact local transforms to guarantee we can snap objects back to grid holes
    cache = {}
    for p in caps + vials:
        cache[p.GetPath().pathString] = UsdGeom.Xformable(p).GetLocalTransformation()

    def apply_defect(prim, tx, ty, tz, lift):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        
        # 1. Anchor it to the exact grid hole
        mat_op = xf.AddTransformOp()
        mat_op.Set(cache[prim.GetPath().pathString])
        
        # 2. Lift it locally (straight up)
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(0, 0, lift))
        
        # 3. Tilt it locally (pivot around its own center)
        r_op = xf.AddRotateXYZOp()
        r_op.Set(Gf.Vec3d(tx, ty, tz))

    for i in range(5):
        print(f"Rendering Variation {i}...", flush=True)
        
        for p in caps + vials:
            UsdGeom.Imageable(p).MakeVisible()
            xf = UsdGeom.Xformable(p)
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            op.Set(cache[p.GetPath().pathString])
            
            damaged_path = p.GetParent().GetPath().pathString + "/DamagedVialDefect"
            damaged_prim = stage.GetPrimAtPath(damaged_path)
            if damaged_prim.IsValid():
                UsdGeom.Imageable(damaged_prim).MakeInvisible()

        if i > 0:
            for parent_path, items in pockets.items():
                cap = items['cap']
                vial = items['vial']
                
                r = random.random()
                if r < 0.08:
                    # Missing Vial
                    if vial: UsdGeom.Imageable(vial).MakeInvisible()
                    if cap: UsdGeom.Imageable(cap).MakeInvisible()
                elif r < 0.16:
                    # Damaged Vial
                    if vial:
                        UsdGeom.Imageable(vial).MakeInvisible()
                        
                        if os.path.exists(a.damaged_vial):
                            damaged_path = parent_path + "/DamagedVialDefect"
                            damaged_prim = stage.GetPrimAtPath(damaged_path)
                            
                            if not damaged_prim.IsValid():
                                damaged_prim = stage.DefinePrim(damaged_path, "Xform")
                                damaged_prim.GetReferences().AddReference(a.damaged_vial)
                            
                            UsdGeom.Imageable(damaged_prim).MakeVisible()
                            new_xf = UsdGeom.Xformable(damaged_prim)
                            new_xf.ClearXformOpOrder()
                            mat_op = new_xf.AddTransformOp()
                            mat_op.Set(cache[vial.GetPath().pathString])
                        else:
                            # Fallback: visually crush existing vial securely in its pocket
                            xf = UsdGeom.Xformable(vial)
                            xf.ClearXformOpOrder()
                            
                            # 1. Base grid hole matrix
                            mat_op = xf.AddTransformOp()
                            mat_op.Set(cache[vial.GetPath().pathString])
                            
                            # 2. Local scale (crush)
                            s_op = xf.AddScaleOp()
                            s_op.Set(Gf.Vec3f(1.0, 1.0, 0.4))
                            
                            # 3. Slide it down slightly to stay on the tray floor
                            t_op = xf.AddTranslateOp()
                            t_op.Set(Gf.Vec3d(0, 0, -0.6))
                    
                    if cap: UsdGeom.Imageable(cap).MakeInvisible()

                elif r < 0.25:
                    # Crooked Assembly
                    tx = random.uniform(10, 20) * random.choice([-1, 1])
                    ty = random.uniform(10, 20) * random.choice([-1, 1])
                    tz = random.uniform(-45, 45)
                    if vial: apply_defect(vial, tx, ty, tz, 2.5)
                    if cap: apply_defect(cap, tx, ty, tz, 2.5)
                else:
                    # Normal vial, crooked cap
                    r2 = random.random()
                    if r2 < 0.12:
                        if cap: UsdGeom.Imageable(cap).MakeInvisible()
                    elif r2 < 0.30:
                        tx = random.uniform(15, 30) * random.choice([-1, 1])
                        ty = random.uniform(15, 30) * random.choice([-1, 1])
                        tz = random.uniform(-90, 90)
                        if cap: apply_defect(cap, tx, ty, tz, 1.5)

        for _ in range(a.subframes + 5):
            kit.update()
        
        current_out = a.out.replace(".png", f"_{i}.png")
        tmp_file = current_out + ".pinhole.png" if cam.get("overscan", 1.0) != 1.0 else current_out
        
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
            
        capture_viewport_to_file(viewport, file_path=tmp_file)
        
        wait_ticks = 0
        while not os.path.exists(tmp_file) and wait_ticks < 200:
            kit.update()
            time.sleep(0.05)
            wait_ticks += 1

        if cam.get("overscan", 1.0) != 1.0:
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp_file,
                                os.path.join(HERE, "cameras", a.camera + ".json"), current_out], check=True)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except subprocess.CalledProcessError:
                os.rename(tmp_file, current_out)
        print(f"wrote {current_out}", flush=True)

except BaseException:
    traceback.print_exc()
    kit.close()
    raise
kit.close()