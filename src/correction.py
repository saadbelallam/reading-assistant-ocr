#!/usr/bin/env python
import os
import subprocess
import argparse
import torch

torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Run the full document correction pipeline on all images in an input directory."
    )
    parser.add_argument("--inputDir", type=str, default="../test/images",
                        help="Directory containing input images")
    parser.add_argument("--outputDir", type=str, default="../test/processed",
                        help="Directory to store final corrected images")
    parser.add_argument("--geoModelPath", type=str, default="preprocessing/models/model_geoNet.pkl",
                        help="Path to pre-trained GeoNet model")
    parser.add_argument("--illModelPath", type=str, default="preprocessing/models/model_illNet.pkl",
                        help="Path to pre-trained illNet model")
    parser.add_argument("--graphcutExe", type=str, default="preprocessing/stitching/Graphcut.exe",
                        help="Absolute path to Graphcut.exe (forward slashes or raw string)")
    parser.add_argument("--tempDir", type=str, default="../test/temp",
                        help="Temporary directory for intermediate files")
    args = parser.parse_args()
    
    # Make sure the specified directories exist
    os.makedirs(args.tempDir, exist_ok=True)
    os.makedirs(args.outputDir, exist_ok=True)
    
    # If your scripts are in DocProj, update these as needed:
    eval_script = os.path.join("preprocessing", "eval.py")
    resampling_script = os.path.join("preprocessing", "resampling.py")
    illum_script = os.path.join("preprocessing", "eval_illumination.py")
    
    # Loop over every image file in the input directory
    for fname in os.listdir(args.inputDir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        base = os.path.splitext(fname)[0]
        input_path = os.path.join(args.inputDir, fname)
        
        # Intermediate file paths in the temp directory
        cropped_path = os.path.join(args.tempDir, base + "_cropped.png")
        flow_path = os.path.join(args.tempDir, base + "_flow.npy")
        stitched_flow_path = os.path.join(args.tempDir, base + "_stitchedFlow.npy")
        rectified_path = os.path.join(args.tempDir, base + "_rectified.png")
        final_path = os.path.join(args.outputDir, base + "_corrected.png")
        
        print(f"\n=== Processing {fname} ===")
        
        # 1) Run eval.py to resize, crop, and estimate patch flows
        eval_cmd = [
            "python3", eval_script,
            "--imgPath", input_path,
            "--modelPath", args.geoModelPath,
            "--saveImgPath", cropped_path,
            "--saveFlowPath", flow_path
        ]
        print("Running eval.py ...")
        subprocess.run(eval_cmd, check=True)
        
        # 2) Run Graphcut.exe to stitch patch flows
        input_flow_abs = os.path.abspath(flow_path).replace("\\", "/")
        output_flow_abs = os.path.abspath(stitched_flow_path).replace("\\", "/")
        graph_cmd = ["wine", args.graphcutExe, input_flow_abs, output_flow_abs]
        print("Running Graphcut.exe ...")
        subprocess.run(graph_cmd, check=True, stdin=subprocess.DEVNULL)
        
        # 3) Run resampling.py to rectify the image
        resample_cmd = [
            "python3", resampling_script,
            "--img_path",cropped_path ,  # or cropped_path if you prefer
            "--flow_path", output_flow_abs,
            "--out_path", rectified_path
        ]
        print("Running resampling.py ...")
        subprocess.run(resample_cmd, check=True)
        
        # 4) Run eval_illumination.py for illumination correction
        illum_cmd = [
            "python3", illum_script,
            "--imgPath", rectified_path,
            "--savPath", final_path,
            "--modelPath", args.illModelPath
        ]
        print("Running eval_illumination.py ...")
        subprocess.run(illum_cmd, check=True)
        
        print(f"Finished processing {fname}. Final corrected image saved to {final_path}")

if __name__ == "__main__":
    main()