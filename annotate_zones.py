import cv2

# Load the captured frame
frame_path = "/path/to/cat-litter-monitor/frame_annotated.jpg"
img = cv2.imread(frame_path)

# Zone definitions
# GREEN rectangle: "Main Litter Box" [700, 540, 1050, 980]
main_box = [700, 540, 1050, 980]
# BLUE rectangle: "Secondary Box" [1050, 680, 1320, 950]
secondary_box = [1050, 680, 1320, 950]

# Colors (BGR format for OpenCV)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
black_color = (0, 0, 0)
white_color = (255, 255, 255)

# Draw rectangles (3px thickness)
cv2.rectangle(img, (main_box[0], main_box[1]), (main_box[2], main_box[3]), green_color, 3)
cv2.rectangle(img, (secondary_box[0], secondary_box[1]), (secondary_box[2], secondary_box[3]), blue_color, 3)

# Function to draw text with dark background
def draw_text_with_bg(img, text, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Background rectangle coordinates
    bg_top_left = (position[0], position[1] - text_height - 8)
    bg_bottom_right = (position[0] + text_width, position[1])
    
    # Draw dark background
    cv2.rectangle(img, bg_top_left, bg_bottom_right, black_color, -1)
    
    # Draw text
    cv2.putText(img, text, (position[0], position[1] - 4), font, font_scale, color, thickness)

# Add labels above each box
draw_text_with_bg(img, "Main Litter Box", (main_box[0], main_box[1]), green_color)
draw_text_with_bg(img, "Secondary Box", (secondary_box[0], secondary_box[1]), blue_color)

# Legend at top-left
legend_x, legend_y = 10, 30
draw_text_with_bg(img, "GREEN = Main Litter Box", (legend_x, legend_y), green_color)
draw_text_with_bg(img, "BLUE = Secondary Box", (legend_x, legend_y + 25), blue_color)

# Save the annotated frame
cv2.imwrite(frame_path, img)
print(f"Annotated frame saved to {frame_path}")
