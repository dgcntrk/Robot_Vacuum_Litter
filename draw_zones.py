from PIL import Image, ImageDraw, ImageFont

# Load the image
input_path = "/path/to/cat-litter-monitor/frame_snapshot.jpg"
output_path = "/path/to/cat-litter-monitor/frame_zones_compared.jpg"

img = Image.open(input_path)
if img.mode != 'RGB':
    img = img.convert('RGB')

draw = ImageDraw.Draw(img)

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Rectangle thickness
THICKNESS = 3

# Try to load a font, fallback to default if not available
try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except:
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_small = ImageFont.load_default()

# SET 1 - Current config zones (RED)
config_zones = [
    ("Config Main", [700, 540, 1050, 980]),
    ("Config Secondary", [1050, 680, 1320, 950]),
]

# SET 2 - Vision-detected zones (GREEN)
detected_zones = [
    ("Detected Box 1", [380, 430, 760, 750]),
    ("Detected Box 2", [820, 560, 1080, 700]),
]

def draw_rectangle_with_label(draw, coords, label, color, font):
    x1, y1, x2, y2 = coords
    # Draw rectangle
    for i in range(THICKNESS):
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)
    # Add label
    label_y = y1 - 25 if y1 > 35 else y1 + 20
    # Draw text with shadow for better visibility
    draw.text((x1+1, label_y+1), label, font=font, fill=BLACK)
    draw.text((x1, label_y), label, font=font, fill=color)

# Draw config zones (RED)
for label, coords in config_zones:
    draw_rectangle_with_label(draw, coords, label, RED, font_small)

# Draw detected zones (GREEN)
for label, coords in detected_zones:
    draw_rectangle_with_label(draw, coords, label, GREEN, font_small)

# Add legend in top-left corner
legend_x = 20
legend_y = 20
line_height = 24

# Draw legend background
legend_bg_coords = [10, 10, 320, 90]
draw.rectangle(legend_bg_coords, fill=(0, 0, 0, 128))

# Legend text
draw.text((legend_x, legend_y), "LEGEND:", font=font_large, fill=WHITE)
draw.text((legend_x, legend_y + line_height), "RED = Current Config", font=font_medium, fill=RED)
draw.text((legend_x, legend_y + 2 * line_height), "GREEN = Vision Detected", font=font_medium, fill=GREEN)

# Save the result
img.save(output_path, quality=95)
print(f"Saved annotated image to: {output_path}")
