import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from scipy.optimize import linear_sum_assignment

out_dir = r"C:\Users\AlShaheen\.gemini\antigravity-ide\scratch\github-profile"

# We want 3 logos on the 300x340 portrait canvas, centered
# Target points: N_POINTS = 900
N_POINTS = 900
W, H = 300, 340

def make_dotnet_logo():
    # Render .NET logo (circle with ".NET" text or clean geometric .NET badge)
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    cx, cy = W // 2, H // 2
    r = 90
    
    # Outer circle with thickness 12
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=12)
    # Inside: Dot and "NET" or modern .NET wave
    # Let's draw modern .NET symbol: a solid dot on the left, and stylized "NET"
    # Or circle with a prominent stylized '</>' or .NET
    # Drawing clean text if font available, else vector curves
    # Dot:
    draw.ellipse([cx - 55, cy + 25, cx - 35, cy + 45], fill=255)
    # 'N':
    draw.line([(cx - 25, cy + 45), (cx - 25, cy - 20)], fill=255, width=10)
    draw.line([(cx - 25, cy - 20), (cx + 5, cy + 45)], fill=255, width=10)
    draw.line([(cx + 5, cy + 45), (cx + 5, cy - 20)], fill=255, width=10)
    # 'E':
    draw.line([(cx + 20, cy + 45), (cx + 20, cy - 20)], fill=255, width=9)
    draw.line([(cx + 20, cy - 20), (cx + 45, cy - 20)], fill=255, width=8)
    draw.line([(cx + 20, cy + 12), (cx + 40, cy + 12)], fill=255, width=8)
    draw.line([(cx + 20, cy + 45), (cx + 45, cy + 45)], fill=255, width=8)
    # 'T':
    draw.line([(cx + 55, cy - 20), (cx + 80, cy - 20)], fill=255, width=8)
    draw.line([(cx + 67, cy - 20), (cx + 67, cy + 45)], fill=255, width=9)
    return im

def make_code_glyph():
    # Render '</>' glyph
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    cx, cy = W // 2, H // 2
    
    # Left bracket '<'
    draw.line([(cx - 40, cy - 65), (cx - 95, cy), (cx - 40, cy + 65)], fill=255, width=14)
    # Slash '/'
    draw.line([(cx + 25, cy - 80), (cx - 25, cy + 80)], fill=255, width=14)
    # Right bracket '>'
    draw.line([(cx + 40, cy - 65), (cx + 95, cy), (cx + 40, cy + 65)], fill=255, width=14)
    return im

def make_docker_logo():
    # Render Docker whale & containers
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    cx, cy = W // 2, H // 2
    
    # Container boxes on whale back
    box_w, box_h = 16, 14
    start_y = cy - 25
    # Row 1 (top): 1 box
    draw.rectangle([cx - 10, start_y - box_h - 4, cx - 10 + box_w, start_y - 4], fill=255)
    # Row 2 (middle): 3 boxes
    for k in [-1, 0, 1]:
        bx = cx + k * (box_w + 4) - 8
        draw.rectangle([bx, start_y, bx + box_w, start_y + box_h], fill=255)
    # Row 3 (bottom containers): 4 boxes
    for k in [-2, -1, 0, 1]:
        bx = cx + k * (box_w + 4) + 2
        draw.rectangle([bx, start_y + box_h + 4, bx + box_w, start_y + 2 * box_h + 4], fill=255)
        
    # Whale body
    whale_pts = [
        (cx - 100, cy + 15),
        (cx + 65, cy + 15),
        (cx + 85, cy + 25),
        (cx + 105, cy + 20),
        (cx + 100, cy + 45),
        (cx + 70, cy + 65),
        (cx - 50, cy + 65),
        (cx - 95, cy + 40),
        (cx - 100, cy + 15)
    ]
    draw.polygon(whale_pts, fill=255)
    # Whale eye
    draw.ellipse([cx + 75, cy + 30, cx + 81, cy + 36], fill=0)
    return im

def sample_n_points(img, n):
    arr = np.array(img)
    ys, xs = np.where(arr > 100)
    pts = np.column_stack([xs, ys])
    if len(pts) < n:
        # duplicate
        idx = np.random.choice(len(pts), n, replace=True)
    else:
        # uniform random subsample
        idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx].astype(np.float64)

p1 = sample_n_points(make_dotnet_logo(), N_POINTS)
p2 = sample_n_points(make_code_glyph(), N_POINTS)
p3 = sample_n_points(make_docker_logo(), N_POINTS)

print(f"Sampled points: {len(p1)}, {len(p2)}, {len(p3)}")

# Optimal transport matching:
# Match p1 -> p2
cost_12 = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
row1, col1 = linear_sum_assignment(cost_12)
p2_matched = p2[col1]

# Match p2_matched -> p3
cost_23 = np.linalg.norm(p2_matched[:, None, :] - p3[None, :, :], axis=2)
row2, col2 = linear_sum_assignment(cost_23)
p3_matched = p3[col2]

print("Optimal transport match complete. Mean distances:")
print(f"p1 -> p2: {cost_12[row1, col1].mean():.2f}")
print(f"p2 -> p3: {cost_23[row2, col2].mean():.2f}")

# Save trajectory points
np.save(os.path.join(out_dir, "traveller_p1.npy"), p1)
np.save(os.path.join(out_dir, "traveller_p2.npy"), p2_matched)
np.save(os.path.join(out_dir, "traveller_p3.npy"), p3_matched)
print("Saved traveller point clouds.")
