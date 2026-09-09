import os
import math
from PIL import Image, ImageDraw

def create_shape_image(shape_type, filename, size=(64, 64), bg_color=(255, 255, 255), fg_color=(0, 0, 0)):
    """
    Draws a specific shape with given foreground and background colors and saves it as an image.
    Supports: circle, square, triangle, cross, diamond, star, hexagon, ring.
    """
    image = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Margins and dimensions
    m = 8
    w, h = size
    cx, cy = w / 2, h / 2
    r = min(w, h) / 2 - m
    
    if shape_type == 'circle':
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg_color)
        
    elif shape_type == 'square':
        draw.rectangle([m, m, w - m, h - m], fill=fg_color)
        
    elif shape_type == 'triangle':
        points = [(cx, m), (m, h - m), (w - m, h - m)]
        draw.polygon(points, fill=fg_color)
        
    elif shape_type == 'cross':
        arm = r * 0.45
        draw.rectangle([cx - arm, m, cx + arm, h - m], fill=fg_color)
        draw.rectangle([m, cy - arm, w - m, cy + arm], fill=fg_color)
        
    elif shape_type == 'diamond':
        points = [(cx, m), (w - m, cy), (cx, h - m), (m, cy)]
        draw.polygon(points, fill=fg_color)
        
    elif shape_type == 'star':
        # 5-pointed star
        points = []
        inner_r = r * 0.45
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            current_r = r if i % 2 == 0 else inner_r
            points.append((cx + current_r * math.cos(angle), cy + current_r * math.sin(angle)))
        draw.polygon(points, fill=fg_color)
        
    elif shape_type == 'hexagon':
        # Regular hexagon
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 6
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=fg_color)
        
    elif shape_type == 'ring':
        # Donut / Ring
        inner_r = r * 0.5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg_color)
        draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=bg_color)
        
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    image.save(filename)

def generate_all_images(output_dir=None):
    """Generates the full dataset of geometric pattern images across multiple color schemes."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = ['circle', 'square', 'triangle', 'cross', 'diamond', 'star', 'hexagon', 'ring']
    colors = {
        'black': (0, 0, 0),
        'red': (220, 38, 38),
        'blue': (37, 99, 235),
        'green': (22, 163, 74),
        'yellow': (234, 179, 8),
        'purple': (147, 51, 234),
        'cyan': (6, 182, 212),
        'orange': (249, 115, 22)
    }
    
    generated_files = []
    for shape in shapes:
        for color_name, color_val in colors.items():
            filename = os.path.join(output_dir, f"{shape}_{color_name}.png")
            create_shape_image(shape, filename, fg_color=color_val)
            generated_files.append(filename)
            
    print(f"Generated {len(generated_files)} sample pattern images ({len(shapes)} shapes x {len(colors)} colors) in {output_dir}")
    return generated_files

if __name__ == "__main__":
    generate_all_images()
