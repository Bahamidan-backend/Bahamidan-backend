import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import scipy.ndimage as ndimage
import os
from scipy.optimize import linear_sum_assignment

base_dir = r"C:\Users\AlShaheen\.gemini\antigravity-ide\scratch\github-profile"
src_photo = r"C:\Users\AlShaheen\.gemini\antigravity-ide\brain\99e245f7-45d7-49dd-b376-009c712574e5\.user_uploaded\media_1790924522416.jpg"

W_GRID, H_GRID = 300, 340
SCALE = 1.20
OFFSET_X = 52.5
OFFSET_Y = 111.5

# 1. Load photo and preprocess
im_raw = Image.open(src_photo).convert("L")
w_orig, h_orig = im_raw.size
target_w = int(h_orig * (W_GRID / H_GRID))
left = (w_orig - target_w) // 2
im_cropped = im_raw.crop((left, 0, left + target_w, h_orig))
im_resized = im_cropped.resize((W_GRID, H_GRID), Image.Resampling.LANCZOS)

# Contrast 1.3x only, autocontrast cutoff=1, UnsharpMask radius=3 percent=140
im_auto = ImageOps.autocontrast(im_resized, cutoff=1)
im_contrast = ImageEnhance.Contrast(im_auto).enhance(1.3)
im_sharp = im_contrast.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
arr = np.array(im_sharp, dtype=np.float32)

# Mask segmentation for Dark mode (clean subject isolation)
raw_mask = (arr > 12)
struct = ndimage.generate_binary_structure(2, 2)
closed_mask = ndimage.binary_closing(raw_mask, structure=struct, iterations=4)
filled_mask = ndimage.binary_fill_holes(closed_mask)
labeled, num_features = ndimage.label(filled_mask)
sizes = ndimage.sum(filled_mask, labeled, range(num_features + 1))
largest_label = sizes[1:].argmax() + 1
subject_mask = (labeled == largest_label)
subject_mask = ndimage.binary_dilation(subject_mask, iterations=2)

# Serpentine Floyd-Steinberg
def floyd_steinberg_serpentine(img_data, mask=None, hard_clear_bleed=True):
    h, w = img_data.shape
    buffer = img_data.copy().astype(np.float64)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        even = (y % 2 == 0)
        xs = range(w) if even else range(w - 1, -1, -1)
        direction = 1 if even else -1
        for x in xs:
            old_val = buffer[y, x]
            if mask is not None and not mask[y, x]:
                new_val = 0.0
                out[y, x] = 0
                err = 0.0 if hard_clear_bleed else (old_val - new_val)
            else:
                new_val = 255.0 if old_val >= 128.0 else 0.0
                out[y, x] = 1 if new_val == 255.0 else 0
                err = old_val - new_val
            if err != 0.0:
                if 0 <= x + direction < w:
                    buffer[y, x + direction] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if 0 <= x - direction < w:
                        buffer[y + 1, x - direction] += err * (3.0 / 16.0)
                    buffer[y + 1, x] += err * (5.0 / 16.0)
                    if 0 <= x + direction < w:
                        buffer[y + 1, x + direction] += err * (1.0 / 16.0)
    return out

dark_dots = floyd_steinberg_serpentine(arr, mask=subject_mask, hard_clear_bleed=True)

# For light mode:
# Create smooth solid mask for the person so ink fills features and background is crisp white
raw_light = (arr > 8)
closed_light = ndimage.binary_closing(raw_light, structure=np.ones((12, 12)), iterations=2)
filled_light = ndimage.binary_fill_holes(closed_light)
labeled_l, num_feat_l = ndimage.label(filled_light)
sizes_l = ndimage.sum(filled_light, labeled_l, range(num_feat_l + 1))
full_subject_mask = (labeled_l == sizes_l[1:].argmax() + 1)
# Smooth boundary
full_subject_mask = ndimage.binary_opening(full_subject_mask, structure=np.ones((4,4)))

light_arr = np.zeros_like(arr)
light_arr[full_subject_mask] = 255.0 - arr[full_subject_mask]
light_dots = floyd_steinberg_serpentine(light_arr, mask=full_subject_mask, hard_clear_bleed=True)

# Convert 2D binary grid to list of horizontal runs
def grid_to_runs(grid):
    runs = [] # list of (y, x_start, length)
    h, w = grid.shape
    for y in range(h):
        in_run = False
        start_x = 0
        for x in range(w):
            if grid[y, x] == 1:
                if not in_run:
                    in_run = True
                    start_x = x
            else:
                if in_run:
                    runs.append((y, start_x, x - start_x))
                    in_run = False
        if in_run:
            runs.append((y, start_x, w - start_x))
    return runs

dark_runs = grid_to_runs(dark_dots)
light_runs = grid_to_runs(light_dots)
print(f"Dark mode runs: {len(dark_runs)}, Light mode runs: {len(light_runs)}")

# Load traveller points
p1 = np.load(os.path.join(base_dir, "traveller_p1.npy"))
p2 = np.load(os.path.join(base_dir, "traveller_p2.npy"))
p3 = np.load(os.path.join(base_dir, "traveller_p3.npy"))
N_TRAV = len(p1)

c1_x, c1_y = p1.mean(axis=0)

# Drift bands
N_BANDS = 94
np.random.seed(42)

def assign_drift_bands(runs, c1_x, c1_y, n_bands=94):
    run_xs = np.array([r[1] + r[2] / 2.0 for r in runs])
    run_ys = np.array([r[0] for r in runs])
    dx = c1_x - run_xs
    dy = c1_y - run_ys
    dist = np.sqrt(dx**2 + dy**2)
    noise = np.random.normal(0, 4.0, size=len(runs))
    metric = dist + noise
    quantiles = np.linspace(0, 100, n_bands + 1)
    thresholds = np.percentile(metric, quantiles)
    thresholds[-1] += 1e-5
    band_indices = np.digitize(metric, thresholds[:-1]) - 1
    band_indices = np.clip(band_indices, 0, n_bands - 1)
    band_dx = np.zeros(n_bands)
    band_dy = np.zeros(n_bands)
    for b in range(n_bands):
        mask = (band_indices == b)
        if mask.any():
            band_dx[b] = dx[mask].mean() * 0.42 * SCALE
            band_dy[b] = dy[mask].mean() * 0.42 * SCALE
    return band_indices, band_dx, band_dy

# Interleaved intro groups
N_INTRO_GROUPS = 60
def assign_intro_groups(runs, n_groups=60):
    # Interleaved hash for uniform temporal shimmering
    run_xs = np.array([r[1] for r in runs])
    run_ys = np.array([r[0] for r in runs])
    groups = ((run_xs * 7 + run_ys * 11) % n_groups).astype(int)
    return groups

dark_bands, dark_bdx, dark_bdy = assign_drift_bands(dark_runs, c1_x, c1_y, N_BANDS)
dark_intro = assign_intro_groups(dark_runs, N_INTRO_GROUPS)

light_bands, light_bdx, light_bdy = assign_drift_bands(light_runs, c1_x, c1_y, N_BANDS)
light_intro = assign_intro_groups(light_runs, N_INTRO_GROUPS)

INFO_ROWS = [
    ("Subject", "Mohammed Bahamidan"),
    ("Role", "Backend Engineer (.NET Developer)"),
    ("Origin", "Mukalla, Yemen"),
    ("Education", "B.S. in IT, Hadhramout Univ"),
    ("Status", "Building + Shipping + Learning"),
    ("ToolChain", "Visual Studio, VS Code, Git, Docker"),
    ("Core.Lang", "C#, SQL, C++"),
    ("Core.Frontend", "HTML5, CSS3, Tailwind, Razor"),
    ("Core.Backend", "ASP.NET Core, EF Core, REST, CQRS"),
    ("Core.Database", "SQL Server, PostgreSQL, SQLite"),
    ("Core.Infra", "Docker, CI/CD Pipelines"),
    ("Grid.Mail", "mohamedsalem230009@gmail.com"),
    ("Grid.LinkedIn", "mohammed-bahamaydan"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.GitHub", "Bahamidan-backend"),
]

def generate_banner_svg(is_dark=True):
    bg_color = "#0A101F" if is_dark else "#F8FAFC"
    border_color = "#1E293B" if is_dark else "#E2E8F0"
    chrome_color = "#22D3EE" if is_dark else "#0891B2"
    text_primary = "#F1F5F9" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    text_dim = "#334155" if is_dark else "#CBD5E1"
    portrait_color = "#A78BFA" if is_dark else "#7C3AED"
    accent_color = "#10B981"
    pill_bg = "#1E293B" if is_dark else "#E2E8F0"
    
    runs = dark_runs if is_dark else light_runs
    bands = dark_bands if is_dark else light_bands
    bdx = dark_bdx if is_dark else light_bdx
    bdy = dark_bdy if is_dark else light_bdy
    intro = dark_intro if is_dark else light_intro
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">')
    svg.append(f'<defs>')
    svg.append(f'  <style>')
    svg.append(f'    @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&amp;display=swap");')
    svg.append(f'    text {{ font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace; }}')
    svg.append(f'    .chrome-label {{ fill: {chrome_color}; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; }}')
    svg.append(f'    .row-label {{ fill: {text_secondary}; font-size: 13px; font-weight: 500; }}')
    svg.append(f'    .row-dots {{ fill: {text_dim}; font-size: 12.5px; }}')
    svg.append(f'    .row-val {{ fill: {text_primary}; font-size: 13px; font-weight: 600; }}')
    svg.append(f'    .live-text {{ fill: #EF4444; font-size: 12px; font-weight: 700; letter-spacing: 1px; }}')
    svg.append(f'    .pill-text {{ fill: {chrome_color}; font-size: 13px; font-weight: 600; }}')
    svg.append(f'  </style>')
    svg.append(f'</defs>')
    
    # Outer terminal window
    svg.append(f'<rect width="1180" height="610" rx="12" fill="{bg_color}" stroke="{border_color}" stroke-width="1.5"/>')
    
    # Title bar
    svg.append(f'<line x1="0" y1="42" x2="1180" y2="42" stroke="{border_color}" stroke-width="1.2"/>')
    svg.append(f'<circle cx="25" cy="21" r="6" fill="#EF4444"/>')
    svg.append(f'<circle cx="45" cy="21" r="6" fill="#F59E0B"/>')
    svg.append(f'<circle cx="65" cy="21" r="6" fill="#10B981"/>')
    svg.append(f'<text x="590" y="26" text-anchor="middle" fill="{text_secondary}" font-size="13" font-weight="500">profile.sh --live</text>')
    
    # Left Frame (VISUAL.MAP)
    svg.append(f'<rect x="30" y="60" width="405" height="495" rx="6" fill="none" stroke="{border_color}" stroke-width="1"/>')
    svg.append(f'<rect x="42" y="52" width="100" height="16" fill="{bg_color}"/>')
    svg.append(f'<text x="50" y="64" class="chrome-label">VISUAL.MAP</text>')
    
    # Portrait container
    svg.append(f'<g transform="translate({OFFSET_X:.1f}, {OFFSET_Y:.1f}) scale({SCALE:.4f})">')
    
    # 1. Intro Shimmer Layer (runs once over ~3.2s)
    svg.append(f'  <!-- INTRO SHIMMER LAYER -->')
    svg.append(f'  <g id="intro-layer">')
    intro_groups_dict = {g: [] for g in range(N_INTRO_GROUPS)}
    for i, r in enumerate(runs):
        intro_groups_dict[intro[i]].append(r)
        
    for g, r_list in intro_groups_dict.items():
        delay = g * (2.0 / N_INTRO_GROUPS)
        path_data = [f"M{x} {y}h{length}" for y, x, length in r_list]
        d_str = "".join(path_data)
        svg.append(f'    <path d="{d_str}" stroke="{portrait_color}" stroke-width="0.95" fill="none" shape-rendering="crispEdges" opacity="0">')
        svg.append(f'      <animate attributeName="opacity" values="0;1" dur="0.35s" begin="{delay:.2f}s" fill="freeze"/>')
        svg.append(f'      <animate attributeName="opacity" values="1;0" dur="0.2s" begin="3.2s" fill="freeze"/>')
        svg.append(f'    </path>')
    svg.append(f'  </g>')
    
    # 2. Main Loop Portrait Layer (~94 Drift Bands)
    svg.append(f'  <!-- LOOP PORTRAIT DRIFT LAYER -->')
    svg.append(f'  <g id="drift-layer" opacity="0">')
    svg.append(f'    <animate attributeName="opacity" values="0;1" dur="0.1s" begin="3.2s" fill="freeze"/>')
    
    band_dict = {b: [] for b in range(N_BANDS)}
    for i, r in enumerate(runs):
        band_dict[bands[i]].append(r)
        
    for b in range(N_BANDS):
        r_list = band_dict[b]
        if not r_list:
            continue
        path_data = [f"M{x} {y}h{length}" for y, x, length in r_list]
        d_str = "".join(path_data)
        dx = bdx[b] / SCALE
        dy = bdy[b] / SCALE
        
        svg.append(f'    <g>')
        svg.append(f'      <animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite" ')
        svg.append(f'        values="0 0; 0 0; {dx:.1f} {dy:.1f}; {dx:.1f} {dy:.1f}; 0 0" ')
        svg.append(f'        keyTimes="0; 0.211; 0.303; 0.908; 1"/>')
        svg.append(f'      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" ')
        svg.append(f'        values="1; 1; 0; 0; 1" ')
        svg.append(f'        keyTimes="0; 0.211; 0.303; 0.908; 1"/>')
        svg.append(f'      <path d="{d_str}" stroke="{portrait_color}" stroke-width="0.95" fill="none" shape-rendering="crispEdges"/>')
        svg.append(f'    </g>')
    svg.append(f'  </g>')
    
    # 3. Travellers Layer (~900 Dots morphing between 3 logos)
    svg.append(f'  <!-- TRAVELLERS SWARM LAYER -->')
    svg.append(f'  <g id="travellers-layer">')
    for i in range(N_TRAV):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        x3, y3 = p3[i]
        
        pos_values = f"{x1:.1f} {y1:.1f}; {x1:.1f} {y1:.1f}; {x1:.1f} {y1:.1f}; {x1:.1f} {y1:.1f}; {x2:.1f} {y2:.1f}; {x2:.1f} {y2:.1f}; {x3:.1f} {y3:.1f}; {x3:.1f} {y3:.1f}; {x1:.1f} {y1:.1f}"
        op_values = "0; 0; 1; 1; 1; 1; 1; 1; 0"
        kt = "0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1"
        
        svg.append(f'    <circle r="1.1" fill="{portrait_color}">')
        svg.append(f'      <animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite" values="{pos_values}" keyTimes="{kt}"/>')
        svg.append(f'      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" values="{op_values}" keyTimes="{kt}"/>')
        svg.append(f'    </circle>')
    svg.append(f'  </g>')
    svg.append(f'</g>') # End portrait container
    
    # Right Side (SYSTEM.INFO readout)
    PANEL_X = 465
    PANEL_Y = 60
    PANEL_W = 685
    PANEL_H = 495
    
    svg.append(f'<rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="6" fill="none" stroke="{border_color}" stroke-width="1"/>')
    svg.append(f'<rect x="{PANEL_X + 12}" y="{PANEL_Y - 8}" width="110" height="16" fill="{bg_color}"/>')
    svg.append(f'<text x="{PANEL_X + 20}" y="{PANEL_Y + 4}" class="chrome-label">SYSTEM.INFO</text>')
    
    # Status badges: LIVE badge + Handle Pill
    live_dot_x = PANEL_X + PANEL_W - 250
    svg.append(f'<circle cx="{live_dot_x}" cy="{PANEL_Y + 20}" r="4" fill="#EF4444">')
    svg.append(f'  <animate attributeName="opacity" values="1;0.2;1" dur="1.4s" repeatCount="indefinite"/>')
    svg.append(f'</circle>')
    svg.append(f'<text x="{live_dot_x + 10}" y="{PANEL_Y + 24}" class="live-text">LIVE</text>')
    
    pill_x = live_dot_x + 55
    pill_w = 175
    svg.append(f'<rect x="{pill_x}" y="{PANEL_Y + 8}" width="{pill_w}" height="24" rx="12" fill="{pill_bg}"/>')
    svg.append(f'<text x="{pill_x + pill_w/2}" y="{PANEL_Y + 24}" text-anchor="middle" class="pill-text">@Bahamidan-backend</text>')
    
    # Table Rows
    start_y = PANEL_Y + 68
    row_spacing = 27.5
    content_w = PANEL_W - 50
    col_left = PANEL_X + 25
    col_right = PANEL_X + PANEL_W - 25
    char_w = 7.8
    
    for idx, (label, val) in enumerate(INFO_ROWS):
        y = start_y + idx * row_spacing
        lbl_w = len(label) * char_w
        val_w = len(val) * char_w
        avail_w = content_w - (lbl_w + val_w) - 16
        n_dots = max(3, int(avail_w / 6.8))
        dots_str = " " + ("·" * n_dots) + " "
        
        svg.append(f'<text x="{col_left}" y="{y}" class="row-label">{label}</text>')
        svg.append(f'<text x="{col_left + lbl_w + 6:.1f}" y="{y}" class="row-dots">{dots_str}</text>')
        svg.append(f'<text x="{col_right}" y="{y}" text-anchor="end" class="row-val" textLength="{val_w:.1f}" lengthAdjust="spacingAndGlyphs">{val}</text>')
        
    # Bottom status bar line
    svg.append(f'<line x1="{PANEL_X + 25}" y1="{PANEL_Y + PANEL_H - 18}" x2="{PANEL_X + PANEL_W - 25}" y2="{PANEL_Y + PANEL_H - 18}" stroke="{border_color}" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{PANEL_X + 25}" y="{PANEL_Y + PANEL_H - 6}" fill="{text_dim}" font-size="11">NODE_ID: YEM-MUK-01 // PROTOCOL: HTTPS // ENCRYPTION: TLS_1.3</text>')
    svg.append(f'<text x="{PANEL_X + PANEL_W - 25}" y="{PANEL_Y + PANEL_H - 6}" text-anchor="end" fill="{accent_color}" font-size="11">● ONLINE</text>')
    
    svg.append(f'</svg>')
    return "\n".join(svg)

dark_svg_str = generate_banner_svg(is_dark=True)
light_svg_str = generate_banner_svg(is_dark=False)

dark_path = os.path.join(base_dir, "dark.svg")
light_path = os.path.join(base_dir, "light.svg")

with open(dark_path, "w", encoding="utf-8") as f:
    f.write(dark_svg_str)

with open(light_path, "w", encoding="utf-8") as f:
    f.write(light_svg_str)

dark_kb = os.path.getsize(dark_path) / 1024
light_kb = os.path.getsize(light_path) / 1024
print(f"Generated polished dark.svg ({dark_kb:.1f} KB) and light.svg ({light_kb:.1f} KB)")
