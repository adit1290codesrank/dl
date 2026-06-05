import math

def generate_segmented_mandala(filename="puzzle_grid_segmented.svg"):
    width, height = 800, 800
    cx, cy = width / 2, height / 2
    
    sectors = 10
    angle_step = 360 / sectors
    
    # Radii spacing 2r, 3r, 4r, 5r, 6r, 7r. Let r=50.
    r_unit = 50
    radii = [r_unit * i for i in range(2, 8)]
    r_inner = radii[0]   # 100
    r_outer = radii[-1]  # 350
    
    # 90 degrees total twist over 5 rings ensures crossings align perfectly
    twist_angle = math.radians(90)
    
    stroke_color = "#1a1a2e"
    stroke_width = "3.5"
    
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="100%" height="100%" fill="#d3d3d3" />',
    ]

    # 1. Draw segmented concentric circles
    # We break each circle into 20 arcs (every 18 degrees) so every intersection is a split point.
    for r in radii[1:-1]: 
        for j in range(20):
            start_angle = math.radians(j * 18 - 90)
            end_angle = math.radians((j + 1) * 18 - 90)
            
            x1 = cx + r * math.cos(start_angle)
            y1 = cy + r * math.sin(start_angle)
            x2 = cx + r * math.cos(end_angle)
            y2 = cy + r * math.sin(end_angle)
            
            # SVG Arc command: A rx ry x-axis-rotation large-arc-flag sweep-flag x y
            path_str = f'M {x1:.3f} {y1:.3f} A {r} {r} 0 0 1 {x2:.3f} {y2:.3f}'
            svg.append(f'<path d="{path_str}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round" />')

    # 2. Draw segmented spiral petals
    num_segments = 5
    steps_per_segment = 30 # high res for smooth curves within the chunk
    
    for i in range(sectors):
        base_angle = math.radians(i * angle_step - 90)
        
        # Break the petal side into 5 distinct sub-paths
        for seg in range(num_segments):
            # t goes from 0 to 1 over the whole petal. 
            # We calculate the start and end t for this specific chunk.
            t_start = seg / num_segments
            t_end = (seg + 1) / num_segments
            
            pts_cw = []
            pts_ccw = []
            
            for step in range(steps_per_segment + 1):
                # Interpolate t within this specific segment
                t = t_start + (t_end - t_start) * (step / steps_per_segment)
                r = r_outer - t * (r_outer - r_inner)
                
                angle_cw = base_angle + (t * twist_angle)
                angle_ccw = base_angle - (t * twist_angle)
                
                x_cw, y_cw = cx + r * math.cos(angle_cw), cy + r * math.sin(angle_cw)
                x_ccw, y_ccw = cx + r * math.cos(angle_ccw), cy + r * math.sin(angle_ccw)
                
                pts_cw.append(f"{x_cw:.3f},{y_cw:.3f}")
                pts_ccw.append(f"{x_ccw:.3f},{y_ccw:.3f}")
                
            # Render the clockwise segment
            svg.append(f'<polyline points="{" ".join(pts_cw)}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round"/>')
            # Render the counter-clockwise segment
            svg.append(f'<polyline points="{" ".join(pts_ccw)}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round"/>')

    # 3. Draw innermost grey circle
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_inner}" fill="#666666" stroke="{stroke_color}" stroke-width="{stroke_width}" />')

    # 4. Input Data Arrays (Outer to Inner)
    arrays = [
        [7, 4, 3, 2, 3, 4, 7, 4, 7, 4], # Outer layer (10 digits)
        [1, 2, 0, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], # Layer 3
        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1], # Layer 2
        [5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0], # Layer 1
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0]  # Inner layer (20 digits)
    ]
    
    def add_text(r_text, angle_deg, text_val):
        if str(text_val) == "0": return # Skip 0
        angle_rad = math.radians(angle_deg)
        tx = cx + r_text * math.cos(angle_rad)
        ty = cy + r_text * math.sin(angle_rad)
        svg.append(
            f'<text x="{tx:.3f}" y="{ty:.3f}" '
            f'font-family="Arial, sans-serif" font-size="22" font-weight="bold" '
            f'fill="red" text-anchor="middle" dominant-baseline="central">{text_val}</text>'
        )

    # 5. Place numbers perfectly centered inside the bounds
    for layer_idx, data in enumerate(arrays):
        layer_num = 4 - layer_idx 
        
        if layer_num == 4:
            r_text = radii[4] + (radii[5] - radii[4]) * 0.35
            for k, val in enumerate(data):
                angle = k * 36 - 90
                add_text(r_text, angle, val)
        else:
            r_text = (radii[layer_num] + radii[layer_num+1]) / 2
            for k, val in enumerate(data):
                angle = k * 18 - 90
                add_text(r_text, angle, val)

    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write("\n".join(svg))
        
    print(f"Generated segmented SVG: {filename}")

if __name__ == "__main__":
    generate_segmented_mandala()