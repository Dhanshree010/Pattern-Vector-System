import os
from PIL import Image, ImageDraw

def create_shape_image(shape_type, filename, size=(64, 64), bg_color=(255, 255, 255), fg_color=(0, 0, 0)):
    """
    Draws a specific shape with given foreground and background colors and saves it as an image.
    """
    image = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Margin
    m = 10
    w, h = size
    
    if shape_type == 'circle':
        draw.ellipse([m, m, w-m, h-m], fill=fg_color)
    elif shape_type == 'square':
        draw.rectangle([m, m, w-m, h-m], fill=fg_color)
    elif shape_type == 'triangle':
        draw.polygon([(w/2, m), (m, h-m), (w-m, h-m)], fill=fg_color)
    elif shape_type == 'cross':
        draw.rectangle([w/2 - m/2, m, w/2 + m/2, h-m], fill=fg_color)
        draw.rectangle([m, h/2 - m/2, w-m, h/2 + m/2], fill=fg_color)
    elif shape_type == 'diamond':
        draw.polygon([(w/2, m), (w-m, h/2), (w/2, h-m), (m, h/2)], fill=fg_color)

    image.save(filename)

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = ['circle', 'square', 'triangle', 'cross', 'diamond']
    colors = {
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0)
    }
    
    for i, shape in enumerate(shapes):
        for j, (color_name, color_val) in enumerate(colors.items()):
            filename = os.path.join(output_dir, f"{shape}_{color_name}.png")
            create_shape_image(shape, filename, fg_color=color_val)
    
    print(f"Generated {len(shapes) * len(colors)} sample images in {output_dir}")
