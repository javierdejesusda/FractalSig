import json
import os

def merge_notebooks():
    demo_path = "notebooks/03_DEMO_Final_End_to_End.ipynb"
    validation_path = "notebooks/03_FractalSig_Validation.ipynb"
    
    with open(demo_path, 'r') as f:
        demo_nb = json.load(f)
        
    with open(validation_path, 'r') as f:
        val_nb = json.load(f)
        
    # Validation cells to append
    val_cells = val_nb['cells']
    
    # We need to inject a bridging cell to map variables
    # Demo uses: real_paths, generated_paths
    # Validation uses: gt_paths, fs_paths
    
    bridge_code = [
        "# --- Bridging Demo Variables to Validation Variables ---\n",
        "print('\\n--- Starting Validation Section ---\\n')\n",
        "try:\n",
        "    if 'real_paths' in locals():\n",
        "        gt_paths = real_paths\n",
        "        print(f\"Using Demo Ground Truth (real_paths): {gt_paths.shape}\")\n",
        "    if 'generated_paths' in locals():\n",
        "        fs_paths = generated_paths\n",
        "        print(f\"Using Demo Generated Paths (generated_paths): {fs_paths.shape}\")\n",
        "except NameError:\n",
        "    print(\"Demo variables not found, will load from disk in validation logic.\")\n"
    ]
    
    bridge_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": bridge_code
    }
    
    # Modify the Validation Data Load cell (Cell index 3 based on view_file, approx)
    # to avoid overwriting if variables exist.
    # Actually, we can just insert the bridge cell at the beginning of the validation cells list.
    # And then in the validation logic, we can wrap the loading code with `if 'gt_paths' not in locals():`
    # transforming the source code of the cell that loads data.
    
    # Let's inspect cells to find the one loading data.
    # Based on view_file: Cell 3 is "1. Load Data" (Code).
    # Its source starts with "# 1. Ground Truth".
    
    modified_val_cells = []
    
    for cell in val_cells:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = "".join(source)
            if "gt_path = DATA_DIR / \"rough_volatility.npy\"" in source_text or "np.load(gt_path)" in source_text:
                # This is the data loading cell.
                # We wrap the loading logic.
                new_source = [
                    "if 'gt_paths' not in locals() or 'fs_paths' not in locals():\n",
                    "    print(\"Loading data from disk (variables not passed from Demo)...\")\n"
                ]
                # Indent original source
                for line in source:
                    new_source.append("    " + line)
                else:
                    new_source.append("\nelse:\n")
                    new_source.append("    print(\"Data already loaded from Demo step.\")\n")
                    
                    # We still need the 'simulate_baseline' function which is defined in that cell!
                    # The original cell defines `simulate_baseline`. If we skip it, we lose the function.
                    # Wait, the cell logic is: Load GT, Load FS, Define simulate_baseline, Run simulate_baseline.
                    # We should ONLY skip the loading parts, but keep the function definition and baseline generation.
                    
                    # Simpler approach: 
                    # Just append the bridge cell. The validation cell re-loading from disk will just overwrite `gt_paths` 
                    # with the same data (from disk), which is fine/harmless, unless the disk data is different.
                    # But the user wants to validate the *Demo's* results (generated_paths).
                    # So rewriting `fs_paths` from disk (pre-computed) would IGNORE the live decoding we just did in Demo!
                    # That is bad. We MUST use `generated_paths`.
                    
                    # So we really should modify the cell to NOT overwrite if exists.
                    pass
                
                # Rewriting the cell content is tricky string manipulation.
                # Let's try to just prepend a check at the top of the cell:
                # "DATA_DIR = Path..." might be needed.
                
                # Let's look at the specific cell content again from memory/view_file.
                # Cell 1 (imports) defines DATA_DIR.
                # Cell 3 (Load Data) uses DATA_DIR.
                
                # Let's Replace the loading block with a conditional block.
                # Original:
                # # 1. Ground Truth
                # try: ... gt_paths = ...
                # # 2. FractalSig Results
                # try: ... fs_paths = ...
                # # 3. Baseline ... def simulate_baseline ...
                
                # We can inject code at the start of this cell to set a flag? 
                # Or just put `gt_paths = real_paths` AFTER this cell?
                # No, if we put it after, the `simulate_baseline(gt_paths)` call inside the cell sees the *loaded* gt_paths. 
                # (Which matches real_paths anyway).
                # But `fs_paths` (FractalSig) loaded from disk might differ from `generated_paths` (Live).
                # The visual audit relies on `fs_paths`.
                
                # Crucial Fix:
                # We will instruct the script to modify the source code string.
                # We will prepend:
                # "global gt_paths, fs_paths\n"
                
                # Let's purely append the cells, but insert a cell *after* the validation loading cell
                # that re-assigns the variables back to the Demo ones?
                # No, because the validation cell *uses* them immediately (e.g. to calculate baseline or print shapes).
                
                # Best approach: Wrap the loading parts.
                # Since replacing exact lines is hard without regex, we will try to make the loading optional.
                
                # Actually, I can just replace the specific lines loading `fs_paths`.
                # "fs_data = np.load(fs_path)"
                
                new_cell_source = []
                # Add a check at the top
                
                # Split original source into lines is consistent.
                
                has_baseline_def = False
                for line in source:
                    if "def simulate_baseline" in line:
                        has_baseline_def = True
                
                if has_baseline_def:
                     # This contains the function def we need.
                     # We'll just construct a cell that defines the function and runs baseline,
                     # but conditionally loads data.
                     
                     new_cell_source.append("# Data Loading & Baseline Generation\n")
                     new_cell_source.append("import numpy as np\n") # ensure numpy is available
                     
                     new_cell_source.append("if 'real_paths' in locals():\n")
                     new_cell_source.append("    gt_paths = real_paths\n")
                     new_cell_source.append("    print(\"Using Demo Ground Truth.\")\n")
                     new_cell_source.append("else:\n")
                     new_cell_source.append("    # Fallback to loading (Original Logic)\n")
                     # ... We would need to copy lines here. It's getting complex.
                     
                     # ALTERNATIVE: 
                     # Just inject `gt_paths = real_paths` and `fs_paths = generated_paths` 
                     # *AFTER* the loading cell, AND *BEFORE* the metric calculation cells.
                     # But wait, the loading cell *also* generates `baseline_paths`.
                     # `baseline_paths = simulate_baseline(gt_paths, ...)`
                     # If we let it load `gt_paths` from disk, `baseline` will be derived from disk GT.
                     # That is fine, GT is GT.
                     # The critical part is `fs_paths` (The Model Output).
                     # We must overwrite `fs_paths` with `generated_paths` BEFORE metrics.
                     
                     # Does the loading cell do anything with `fs_paths`?
                     # It prints its shape.
                     # "print(f\"Loaded FractalSig: {fs_paths.shape}\")"
                     # It doesn't seem to calculate metrics on it yet. 
                     # Metrics are in Cell 6 `calculate_increment_std(fs_paths)`.
                     
                     # SO:
                     # 1. Append all Validation cells.
                     # 2. Insert a "Override" cell AFTER the data loading cell (Cell 3 of val)
                     #    that sets `fs_paths = generated_paths` (if available).
                     #    And also `gt_paths = real_paths`.
                     # 3. But wait, `baseline_paths` is computed in Cell 3 using the *just loaded* `gt_paths`.
                     #    If `gt_paths` (disk) == `real_paths` (demo), then `baseline_paths` is correct.
                     #    They should be identical (same .npy file).
                     
                     pass
            
        modified_val_cells.append(cell)

    # Let's insert the override cell AFTER the loading cell (which is roughly the 2nd code cell in val_cells)
    # Cell 0: Markdown
    # Cell 1: Code (Imports)
    # Cell 2: Markdown
    # Cell 3: Code (Load Data + Baseline)
    
    # We want to override fs_paths AFTER Cell 3 so that subsequent cells use the Live Demo paths.
    
    override_code = [
        "\n# --- INTEGRATION OVERRIDE ---\n",
        "# Use the Live Demo paths instead of the stored file for validation metrics\n",
        "if 'generated_paths' in locals():\n",
        "    fs_paths = generated_paths\n",
        "    print(f\"\\n[Integration] Overwrote 'fs_paths' with Live Demo results: {fs_paths.shape}\")\n",
        "else:\n",
        "    print(\"\\n[Integration] Live Demo paths not found, using loaded file.\")\n",
        "\n",
        "if 'real_paths' in locals():\n",
        "    gt_paths = real_paths\n",
        "    # Ensure baseline is computed on the correct GT if needed, though they should be same.\n"
    ]
    
    override_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": override_code
    }
    
    # Insert override after Cell 3
    # Note: verify indices logic
    # val_cells[0] is Markdown header
    # val_cells[1] is Imports code
    # val_cells[2] is Markdown
    # val_cells[3] is Data Load code
    
    # We insert at index 4
    val_cells.insert(4, override_cell)
    
    # Concatenate
    demo_nb['cells'].extend(val_cells)
    
    # Save
    with open(demo_path, 'w') as f:
        json.dump(demo_nb, f, indent=1)
    
    print("Successfully merged notebooks with variable integration.")

if __name__ == "__main__":
    merge_notebooks()
